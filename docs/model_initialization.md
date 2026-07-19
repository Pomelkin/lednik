# Model Initialization

This guide explains how to create a **student** model from a pretrained **teacher**
transformer. Initialization is the first step of every distillation run: it produces a
small model whose embedding table is already seeded with knowledge extracted from the
teacher, so distillation training starts from a sensible point instead of random weights.

Lednik ships two student architectures:

| Student | Class | What it is | When to use it |
| --- | --- | --- | --- |
| **Static Embeddings** | [`StaticEmbeddingsModel`](../lednik/models/modeling_static.py) | A lookup table (`vocab_size × hidden_size`) plus per-token weights, RMSNorm and mean pooling. No attention, no context. | Extremely low-latency / CPU-only inference where contextualization is not required. |
| **Lednik Transformer** | [`LednikModel`](../lednik/models/modeling_lednik.py) | A tiny encoder built from an explicit per-layer stack: `full-attention` blocks (RoPE, optional attention gating, SwiGLU/GeGLU MLP), bidirectional `gated-delta-net` blocks (linear attention), and experimental `moba` blocks. Flash-Attention / varlen SDPA fast paths. | When you need contextual embeddings but a fraction of the teacher's size. Mix in `gated-delta-net` layers for linear scaling with sequence length. |

Both students are standard `transformers.PreTrainedModel` subclasses, so `from_pretrained`
/ `save_pretrained` work as usual. They also mix in
`LightningCheckpointLoaderMixin` from the [`kostyl`](#about-kostyl) toolkit, which adds
`from_lightning_checkpoint(...)` for loading raw PyTorch Lightning `.ckpt` files (see
[Training with ClearML](./training_with_clearml.md)).

---

## How initialization works

Both factory functions follow the same recipe:

```
teacher transformer
        │
        ▼
extract per-token embeddings   (lednik.emb_utils.extract_embeddings)
  run every vocab token through the teacher and pool the hidden states
        │
        ▼
PCA to the student hidden size (lednik.initialization.dim_reduction.PCA)
        │
        ▼
seed the student's embedding table with the reduced vectors
        │
        ▼
StaticEmbeddingsModel  /  LednikModel
```

- `pooling` selects how each token's hidden states are pooled into a single vector
  during extraction: `"mean"`, `"cls"`, or `"last"`
  (see [`get_sentence_embedding`](../lednik/emb_utils.py)).
- PCA reduces the teacher's hidden size to the student's `hidden_size`/`embedding_dim`.
  The explained-variance ratio is logged so you can judge how much information survives
  the projection.
- The whole routine runs under a temporary default dtype and restores it afterwards.

---

## Prerequisites

```bash
# 1. Clone with the kostyl submodule (or initialize it after the fact)
git submodule update --init --recursive

# 2. Install dependencies (uv is the project's package manager)
uv sync                       # core + default groups (dev, distill)
uv sync --group flash-attn    # optional: Flash-Attention for LednikModel on GPU
```

`kostyl-toolkit[ml]` is a normal dependency declared in [`pyproject.toml`](../pyproject.toml);
the `kostyl_toolkit/` submodule is the editable source of that same library (see
[About kostyl](#about-kostyl)).

---

## 1. Load a teacher

Any encoder that returns `last_hidden_state` works. The teacher and the student **share
the teacher's tokenizer** — do not retrain or swap the tokenizer before distillation.

```python
import torch
from transformers import AutoModel, AutoTokenizer

teacher = AutoModel.from_pretrained("deepvk/USER-bge-m3").to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-bge-m3")
```

---

## 2a. Initialize a Static Embeddings model

```python
from lednik.initialization.factory import create_static_embeddings_model

static_model = create_static_embeddings_model(
    model=teacher,
    tokenizer=tokenizer,
    embedding_dim=300,                    # student hidden size after PCA
    pooling="mean",                       # how to pool teacher hidden states
    embedding_extraction_batch_size=256,
    dtype="float32",                      # str or torch.dtype
    modify_tokenizer=False,               # keep teacher tokenization for distillation
    sif_coefficient=1e-4,                 # SIF token weighting; None -> uniform weights
    output_device=None,                   # defaults to teacher's device
)

static_model.save_pretrained("weights/static_base")
tokenizer.save_pretrained("weights/static_base")
```

### `create_static_embeddings_model` parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `PreTrainedModel` | — | Teacher to extract embeddings from. |
| `tokenizer` | `TokenizersBackend` | — | Teacher tokenizer. |
| `embedding_dim` | `int` | — | Target embedding dimensionality (PCA components). |
| `pooling` | `"mean" \| "last" \| "cls"` | — | Pooling for extraction. |
| `embedding_extraction_batch_size` | `int` | — | Batch size while sweeping the vocab. |
| `dtype` | `str \| torch.dtype` | `"float32"` | Compute/output dtype. |
| `modify_tokenizer` | `bool` | `False` | Customize the tokenizer (normalizer + post-processor). Keep `False` for distillation; customize **after** training via [`customize_tokenizer`](../lednik/initialization/tokenizer_utils.py). |
| `sif_coefficient` | `float \| None` | `1e-4` | Smooth Inverse Frequency token weighting. `None` ⇒ uniform weights. |
| `output_device` | `torch.device \| None` | `None` | Device for the returned model. Defaults to the teacher's device. |
| `**kwargs` | `Any` | — | Forwarded to [`StaticEmbeddingsConfig`](../lednik/models/configuration_static.py). |

The model stores two embedding tables — `embeddings` (the distilled vectors) and
`token_pos_weights` (SIF weights). Its `forward` mean-pools the masked, weighted,
RMSNormed token vectors into a `sentence_embeddings` tensor.

> **Note on `modify_tokenizer`.** A customized tokenizer is **required** before you can
> build a `StaticEmbeddingsForSequenceClassification` head (the config flag
> `is_tokenizer_customized` must be `True`). For plain embedding distillation, leave it
> `False`.

---

## 2b. Initialize a Lednik Transformer

A `LednikModel` needs a `LednikConfig`. **`rope_parameters` is mandatory** — the rotary
embedding raises if it is `None`.

```python
from lednik.models import LednikConfig
from lednik.initialization.factory import create_lednik_transformer

config = LednikConfig(
    hidden_size=384,
    num_attention_heads=6,
    intermediate_size=1152,
    max_position_embeddings=1024,
    hidden_act="silu",            # "silu" -> SwiGLU MLP, "gelu" -> GeGLU MLP
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, # the layer stack is explicit; mix attention kinds freely:
    layers=["full-attention", "gated-delta-net", "full-attention"], # vocab_size / pad_token_id are auto-aligned to the tokenizer below
)

lednik_model = create_lednik_transformer(
    model=teacher,
    tokenizer=tokenizer,
    model_config=config,
    pooling="mean",
    embedding_extraction_batch_size=256,
    dtype="float32",
    output_device=None,
)

lednik_model.save_pretrained("weights/lednik_base")
tokenizer.save_pretrained("weights/lednik_base")
```

### `create_lednik_transformer` parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `PreTrainedModel` | — | Teacher. |
| `tokenizer` | `TokenizersBackend` | — | Teacher tokenizer. |
| `model_config` | `LednikConfig` | — | Student architecture. PCA targets `model_config.hidden_size`. |
| `pooling` | `"mean" \| "last" \| "cls"` | — | Pooling for extraction. |
| `embedding_extraction_batch_size` | `int` | — | Batch size while sweeping the vocab. |
| `dtype` | `str \| torch.dtype` | `"float32"` | Compute/output dtype. |
| `output_device` | `torch.device \| None` | `None` | Device for the returned model. |

`vocab_size` and `pad_token_id` in the config are automatically reconciled with the
tokenizer (a warning is logged if they differ and the config is updated in place).

### Key `LednikConfig` fields

See [`configuration_lednik.py`](../lednik/models/configuration_lednik.py) for the full list.

| Field | Default | Notes |
| --- | --- | --- |
| `vocab_size` | `30522` | Overwritten to match the tokenizer by the factory. |
| `hidden_size` | `384` | Auto-rounded up to a multiple of 8. |
| `output_hidden_size` | `None` | If set, a final `Linear(hidden_size → output_hidden_size)` is added; sentence-embedding dim becomes this. |
| `layers` | `[]` | **The layer stack.** A list of `"full-attention"` / `"gated-delta-net"` / `"moba"` entries; `num_hidden_layers` is derived from its length. |
| `num_attention_heads` / `head_dim` | `6` / `64` | For full-attention blocks; `head_dim` is auto-rounded up to a multiple of 8. |
| `use_gated_attention` | `True` | Output gating for full-attention blocks. |
| `gdn_*` | — | Gated DeltaNet block parameters: `gdn_bidir`, `gdn_num_heads`, `gdn_head_dim`, `gdn_expand_v`, `gdn_use_short_conv`, `gdn_conv_size`, `use_mlp_after_gdn`, … |
| `moba_chunk_size` / `moba_topk` | `32` / `3` | MoBA block parameters. |
| `intermediate_size` | `576` | MLP hidden dim; auto-rounded up to a multiple of 8. |
| `hidden_act` | `"silu"` | `"silu"` ⇒ `LigerSwiGLUMLP`, `"gelu"` ⇒ `LigerGEGLUMLP`. |
| `max_position_embeddings` | `1024` | RoPE max length; also the serving truncation limit. |
| `rope_parameters` | `None` | **Required.** e.g. `{"rope_type": "default", "rope_theta": 10000.0}`. |
| `embedding_dropout` / `attention_dropout` / `out_attn_dropout` / `mlp_dropout` | `0.0` | Regularization knobs (typically set during training, not at init). |

---

## 3. Inference

### Static Embeddings

```python
import torch

enc = tokenizer(["привет мир", "hello world"], padding=True, return_tensors="pt")
enc = {k: v.to(static_model.device) for k, v in enc.items()}

with torch.inference_mode():
    out = static_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

sentence_embeddings = out.sentence_embeddings   # (batch, hidden_size)
token_embeddings = out.token_embeddings         # (batch, seq, hidden_size)
```

`StaticEmbeddingsModel.forward` returns a
[`StaticEmbeddingsOutput`](../lednik/models/outputs.py) with `token_embeddings`,
`sentence_embeddings` and `pos_weights`.

### Lednik Transformer

```python
import torch

# For CPU / no Flash-Attention, load with eager attention:
# lednik_model = LednikModel.from_pretrained("weights/lednik_base", attn_implementation="eager")

enc = tokenizer(["привет мир", "hello world"], padding=True, return_tensors="pt")
enc = {k: v.to(lednik_model.device) for k, v in enc.items()}

with torch.inference_mode():
    out = lednik_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

sentence_embeddings = out.sentence_embeddings   # (batch, output_hidden_size or hidden_size)
last_hidden_state = out.last_hidden_state        # (batch, seq, ...)
```

`LednikModel.forward` returns a
[`LednikModelOutput`](../lednik/models/outputs.py) with `last_hidden_state` and
mean-pooled `sentence_embeddings`.

> **Attention backend.** `LednikModel` auto-selects `flash_attention_4` /
> `flash_attention_2` when available, otherwise falls back. Flash-Attention and the
> `sdpa` (torch varlen) backends run an **unpadded** path internally and need
> `cu_seqlens` / `max_seqlen`, which the model derives from `attention_mask`. For simple
> CPU runs pass `attn_implementation="eager"`.

For benchmarking either model on MTEB, the repo provides a ready adapter,
[`MTEBModelWrapper`](../bench/mteb_testing/model_wrapper.py); see
[Training with ClearML → Benchmarking](./training_with_clearml.md#benchmarking-with-mteb).
For loading trained checkpoints and serving, see [Usage](./usage.md).

---

## Saving and reloading

```python
# Save (HF format) — a directory with config.json + model.safetensors
model.save_pretrained("weights/my_student")
tokenizer.save_pretrained("weights/my_student")

# Reload with a concrete class...
from lednik.models import LednikModel, StaticEmbeddingsModel
model = LednikModel.from_pretrained("weights/my_student")

# ...or let AutoLednikModel resolve the class from the checkpoint config
from lednik.models import AutoLednikModel
model = AutoLednikModel.from_pretrained("weights/my_student")

# Lightning .ckpt files from distillation runs work too (weights live under
# the "student." prefix inside the training module's state dict):
model = AutoLednikModel.from_pretrained(
    "checkpoints/last.ckpt", weights_prefix="student.", strict_prefix=True
)
```

Every model/config class is registered via the `@register_model` / `@register_config`
decorators in [`lednik/models/auto.py`](../lednik/models/auto.py);
[`AutoLednikModel`](../lednik/models/auto.py) resolves the concrete class from a
checkpoint's `architectures` through these registries — this is what the ClearML pipeline,
the serving stack and the benches use to load any Lednik checkpoint. The companion
`is_lednik_checkpoint(path)` distinguishes Lednik checkpoints from plain Transformers ones.

---

## Next steps

Once you have an initialized student you can distill it:

- **No ClearML** (build everything in Python): [Training without ClearML](./training_without_clearml.md)
- **With ClearML** (config sync, artifact loading, remote queues, online validation):
  [Training with ClearML](./training_with_clearml.md)

---

## About kostyl

`kostyl` (a.k.a. **kostyl-toolkit**) is the author's personal ML toolkit, vendored as the
[`kostyl_toolkit/`](../kostyl_toolkit) git submodule and also installed as the
`kostyl-toolkit[ml]` dependency. It is used throughout Lednik for logging, config schemas,
optimizer/scheduler factories, distributed helpers, PyTorch-Lightning glue and the entire
ClearML integration. Wherever you see an import from `kostyl.*`, it comes from this
submodule. The pieces relevant to each workflow are documented inline in the two training
guides.
