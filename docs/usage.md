# Usage

This guide covers using trained Lednik models: loading checkpoints and resolving model
references (general-purpose utilities useful in any inference code), and serving — the
LitServe embedding server, its request protocol and scaling knobs, and Docker deployment.

---

## Loading models

### `AutoLednikModel`

Use `AutoLednikModel` whenever you need to load a trained checkpoint without hardcoding its
concrete class — the same call works for a `LednikModel` and a `StaticEmbeddingsModel`, for
an HF-format directory and a Lightning `.ckpt` from a distillation run:

```python
from lednik.models import AutoLednikModel

# HF-format directory (config.json + model.safetensors)
model = AutoLednikModel.from_pretrained("weights/lednik_base")

# Lightning .ckpt from a distillation run: weights are stored under the
# "student." prefix inside the training module's state dict
model = AutoLednikModel.from_pretrained(
    "checkpoints/epoch=3-step=1317.ckpt",
    weights_prefix="student.",
    strict_prefix=True,
)
```

Under the hood the concrete class is resolved from the checkpoint's config via the
`@register_model` / `@register_config` registries
([`lednik/models/auto.py`](../lednik/models/auto.py)).

When a script has to accept *any* model — Lednik students and plain Transformers teachers
alike — use `is_lednik_checkpoint` to pick the loader:

```python
from transformers import AutoModel
from lednik.models import AutoLednikModel, is_lednik_checkpoint

if is_lednik_checkpoint(path):
    model = AutoLednikModel.from_pretrained(path)
else:
    model = AutoModel.from_pretrained(path)
```

(Detection: a `.ckpt` file is always Lednik; a directory is matched by its `architectures`
against the registry.) This is exactly how the server and the forward/MTEB benches load
their `--model` argument.

### `determine_path`

Use [`lednik.path_utils.determine_path`](../lednik/path_utils.py) when a CLI or config
takes a model/tokenizer reference and you don't want to care where the artifact lives —
the user can pass a ClearML model ID, an HF Hub repo id, or a local path, and you get back
a validated local path either way:

```python
from lednik.path_utils import determine_path

model_path = determine_path(args.model, is_tokenizer=False)
tokenizer_path = determine_path(args.tokenizer, is_tokenizer=True)
```

Resolution is tried in order: ClearML model ID (when `clearml` is installed, fetched into
the ClearML cache) → HF Hub repo id (`snapshot_download`) → local path as-is. The result is
validated for content (`config.json` / `.ckpt` for models, `tokenizer.json` for
tokenizers). The server and every benchmark CLI pass their `--model` / `--tokenizer`
through it, which is why all three reference forms work everywhere.

---

## The embedding server

[`lednik/serving/server.py`](../lednik/serving/server.py) implements `LednikServer`, a
[LitServe](https://github.com/Lightning-AI/LitServe) `LitAPI`:

- **`setup`** loads the model (via `is_lednik_checkpoint` → `AutoLednikModel`/`AutoModel`)
  and tokenizer on a CUDA device, picks bf16/fp16, clamps `max_seq_length` to the model
  limit, and configures the Rust backend tokenizer (`no_padding` +
  `enable_truncation(max_seq_length)` so truncation preserves the final special token).
- **`decode_request`** validates the payload with a pydantic model (`EmbedRequest`);
  litserve passes the annotation to FastAPI, so invalid requests get a proper 422 in the
  API process before touching the workers.
- **`batch`** collates requests: raw texts are tokenized in one `encode_batch_fast` call,
  sequences are truncated and padded **to the batch maximum** (rounded up to a multiple
  of 8, not to `max_seq_length` — keeping the varlen profile), and the attention mask is
  built server-side.
- **`predict`** runs the forward pass and returns L2-normalized, float32 CPU sentence
  embeddings (CUDA tensors must not cross the response queue).
- **`encode_response`** serializes one embedding per request.

### Request protocol

`POST /predict` with exactly one of two fields:

```jsonc
{"token_ids": [0, 1234, 567, 2]}   // pre-tokenized (client-side tokenization)
{"text": "Как оформить возврат?"}  // raw text (server-side tokenization)
```

Response: `{"embedding": [ ... ]}`. Sending both fields, neither, or an empty sequence is
a 422.

The `token_ids` path exists for benchmarking: it moves tokenization out of the server so
load tests measure the model, not the tokenizer. **The client must use exactly the same
tokenizer the server was started with** — this is not validated. The `text` path is the
convenient production-style entry.

> The tokens-only protocol replaced an earlier OpenAI-compatible
> (`/v1/embeddings`, `OpenAIEmbeddingSpec`) version — see git history if you need it back.

### CLI and scaling knobs

```bash
uv run python -m lednik.serving.server \
  --model <path|clearml-id|hf-repo> \
  --tokenizer <path|clearml-id|hf-repo> \
  --max-batch-size 512 --batch-timeout 0.001 \
  --num-workers 4 --num-api-servers 4 --fast-queue true \
  --max-seq-length 8192 --devices 1
```

| Option | What it scales |
| --- | --- |
| `--max-batch-size` / `--batch-timeout` | Dynamic batching: how many queued requests one forward consumes, and how long a worker waits to fill a batch. |
| `--num-workers` | Inference worker processes per GPU. Multiple workers split the request stream (smaller batches each) and only pay off for small models with idle SMs — pair with CUDA MPS, otherwise contexts time-slice. |
| `--num-api-servers` | Parallel HTTP/API processes over the shared queue. This is the knob for the Python-side ceiling (HTTP parsing, pydantic, JSON serialization). |
| `--fast-queue` | ZMQ transport between API processes and workers instead of multiprocessing manager queues (an RPC per queue op). |
| `--max-seq-length` | Truncation limit; clamped to the model's `max_position_embeddings`. |

### Docker deployment

The `serving` compose service builds [`docker/serving.Dockerfile`](../docker/serving.Dockerfile)
and is configured through `.env`:

```bash
SERVING_MODEL=...            # path | ClearML ID | HF repo
SERVING_TOKENIZER=...
SERVING_MAX_BATCH_SIZE=512
SERVING_BATCH_TIMEOUT=0.001
SERVING_NUM_WORKERS=4
SERVING_NUM_API_SERVERS=4
SERVING_FAST_QUEUE=true
SERVING_MAX_SEQ_LENGTH=8192
```

```bash
docker compose --profile serving build serving
docker compose --profile serving up -d
```

Notes:

- `/tmp/nvidia-mps` is mounted into the container: if a CUDA MPS daemon runs on the host
  (`nvidia-cuda-mps-control -d`), workers transparently attach to it, enabling truly
  concurrent kernels from multiple workers. Verify with `nvidia-smi` (an
  `nvidia-cuda-mps-server` process appears).
