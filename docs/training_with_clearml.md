# Training with ClearML

This guide covers the **production distillation pipeline** in
[`pipelines/distill/`](../pipelines/distill). It runs the same
`DistillationModule`/`ContrastiveCollator` described in
[Training without ClearML](./training_without_clearml.md), and adds everything
[ClearML](https://clear.ml/) gives you:

- **Config syncing** — the YAML config is connected to the ClearML task and can be
  overridden from the UI for remote runs.
- **Artifact loading** — the teacher config, the student and the tokenizer are pulled
  from the ClearML model registry by ID; datasets are pulled from ClearML Datasets.
- **Checkpoint uploading** — improved checkpoints are pushed back to the registry as new
  model versions.
- **Remote execution** — enqueue the run on a ClearML agent queue with one flag.
- **Online validation** — embedding-quality metrics (KNN / LogReg / MRR) computed
  outside the training loop: by a **separate worker** consuming a Redis stream, or
  **locally in-process** when no Redis is configured.

Almost all ClearML/Lightning glue comes from the
[`kostyl`](./model_initialization.md#about-kostyl) submodule
(`kostyl.ml.integrations.clearml`, `kostyl.ml.integrations.lightning`).

---

## Pipeline at a glance

```
                 configs/training_settings.yaml
                              │  connect_as_file (ClearML config sync)
                              ▼
   ClearML registry ──► teacher config (InputModel: hidden_size only)
        │              student weights (AutoLednikModel)
        │              tokenizer (load_tokenizer_from_clearml)
        ▼
   ClearML Datasets ─► DataModule (download; glob train/ val/; concat)
        │                 │  ContrastiveCollator (+ typo augmentation)
        ▼                 ▼
   DistillationModule (Lightning) ──fit──► checkpoints ──► ClearML model registry
        │                                                  (new versions, best only)
        └── val embeddings ──► EvaluationDispatcher
                                 ├─ Redis stream ──► EvaluationWorker (separate process/host)
                                 └─ no Redis / Redis down ──► local EvaluationRunner
                                                 both log KNN/LogReg/MRR to the task
```

Entry point: [`pipelines/distill/run.py`](../pipelines/distill/run.py).

---

## 1. Prerequisites

```bash
git submodule update --init --recursive
uv sync                          # includes the `distill` group (clearml[s3], lightning, ...)
uv sync --group flash-attn       # optional, for LednikModel on GPU
```

Configure ClearML credentials once (writes `~/clearml.conf`):

```bash
uv run clearml-init
```

For S3-backed artifact storage, the `clearml[s3]` extra is already pulled in by the
`distill` group; configure your bucket in `clearml.conf`.

---

## 2. The dataset format

The pipeline trains on **query–positive pairs** (optionally with hard negatives) where
teacher sentence embeddings are **precomputed offline** — the teacher never runs during
training. This section is the contract; it applies to the no-ClearML path as well, where
you feed the same rows through a `DataLoader` yourself.

### 2.1 Row schema (what the collator reads)

[`ContrastiveCollator`](../lednik/distill/collator.py) reads each row by column name;
the names are configurable (`data:` section, [§4](#4-configuration)). Per row:

| Logical field                                    | Config key                          | Type                                     | Notes                                                              |
| ------------------------------------------------ | ----------------------------------- | ---------------------------------------- | ------------------------------------------------------------------ |
| Query tokens                                     | `query_tok_colname`               | `list[int]` or `list[list[int]]`     | Token ids**without special tokens**.                         |
| Query text                                       | `query_text_colname`              | `str` or `list[str]`                 | Raw text of the same query; used for augmentation.                 |
| Query teacher embedding                          | `query_teacher_embedding_colname` | `list[float]` or `list[list[float]]` | Teacher sentence vector of the clean text.                         |
| Positive tokens                                  | `pos_tok_colname`                 | same as query                            | —                                                                 |
| Positive text                                    | `pos_text_colname`                | same as query                            | —                                                                 |
| Positive teacher embedding                       | `pos_teacher_embedding_colname`   | same as query                            | —                                                                 |
| Negative tokens / text / embedding*(optional)* | `neg_*_colname`                   | same as query                            | Hard negatives. All three keys must be set together or not at all. |
| Label*(optional, val only)*                    | `val_label_colname`               | `int` or `list[int]`                 | Class label for the KNN / LogReg validation metrics.               |

Rules the collator enforces (it raises on violations):

- **No special tokens in the ids.** The collator appends them itself: `[CLS] … [SEP]`
  when the tokenizer has cls/sep, otherwise `… [EOS]`, truncating to `max_len`
  (defaults to `tokenizer.model_max_length`).
- **Variants.** A field may hold several variants (`list[list[int]]` + `list[str]` +
  `list[list[float]]`, e.g. paraphrases). Tokens, text and embedding must have the
  **same number of variants**; one index is sampled per row per epoch, and the same
  index is used for all three.
- **Embedding width** must equal the teacher's hidden size (the module builds a
  `student → teacher` projection from it).

A minimal single-variant row (teacher dim 1024):

```python
row = {
    "query":           "как оформить возврат товара",
    "query-tokens":    [4211, 8765, 1289, 6710],        # no special tokens
    "query-embedding": [0.0123, -0.4210, ...],          # len == 1024
    "pos":             "процедура возврата покупки",
    "pos-tokens":      [5012, 990, 15672],
    "pos-embedding":   [0.0981, 0.0204, ...],
    "label": 3,                                          # optional, val split only
}
```

### 2.2 Typo augmentation (why text columns are required)

With `data.aug_prob > 0`, that fraction of training rows is corrupted with realistic
typos ([SAGE](https://github.com/ai-forever/sage) `SBSCCorruptor`, ≥5 typos) and
**re-tokenized on the fly**; the target stays the teacher embedding of the **clean**
text. The student learns to embed noisy input into clean-text vectors. Validation
always runs with `aug_prob = 0`. Texts that repeatedly fail corruption are blacklisted
for the rest of the run and used as-is.

### 2.3 On-disk layout

Each dataset is a directory with `train/` and `val/` subfolders, each an HF
`datasets.Dataset` saved with `save_to_disk`:

```
<dataset_root>/
├── train/        # datasets.save_to_disk output (*.arrow, dataset_info.json, state.json)
└── val/
```

### 2.4 Building a dataset from raw pairs

End to end: tokenize without special tokens, precompute teacher embeddings, save both
splits.

```python
import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from lednik.emb_utils import get_sentence_embedding

teacher = AutoModel.from_pretrained("deepvk/USER-bge-m3").to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-bge-m3")

pairs = [("как оформить возврат товара", "процедура возврата покупки"), ...]

@torch.inference_mode()
def embed(texts: list[str]) -> list[list[float]]:
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
    out = teacher(**enc)
    sent = get_sentence_embedding(
        out.last_hidden_state, enc["attention_mask"], pooling_method="mean"
    )
    return sent.float().cpu().tolist()

def tokenize(texts: list[str]) -> list[list[int]]:
    return tokenizer(texts, add_special_tokens=False)["input_ids"]

queries, positives = map(list, zip(*pairs))
ds = Dataset.from_dict({
    "query": queries,
    "query-tokens": tokenize(queries),
    "query-embedding": embed(queries),
    "pos": positives,
    "pos-tokens": tokenize(positives),
    "pos-embedding": embed(positives),
})
split = ds.train_test_split(test_size=0.05, seed=42)
split["train"].save_to_disk("my_dataset/train")
split["test"].save_to_disk("my_dataset/val")
```

Batch the `embed` calls for real corpora. Match the pooling method to how you intend
the teacher's embeddings to be used; `USER-bge-m3` uses CLS pooling in its model card,
`mean` is what the repo's datasets were built with.

### 2.5 How the `DataModule` consumes it

[`pipelines/distill/datamodule.py`](../pipelines/distill/datamodule.py), given the
`data.datasets` name → ClearML Dataset ID mapping:

1. downloads all datasets in parallel (`prepare_data`);
2. globs `**/train/` and `**/val/` inside each download, so nesting depth doesn't
   matter;
3. loads each split with `load_from_disk`, keeps **only** the configured columns, and
   skips (with a warning) datasets that lack them — you can mix datasets with different
   extra columns;
4. concatenates everything into one train and one val dataset.

Both dataloaders use `drop_last=True` — required, because the contrastive loss aligns
query `i` with positive `i` and needs equal-sized blocks.

---

## 3. Registering artifacts in ClearML

The pipeline references three models and N datasets by ClearML ID. Registering is a
one-time step per artifact.

### 3.1 Teacher

Only `config_dict["hidden_size"]` of the teacher entry is read (to size the
student→teacher projection); teacher weights are never downloaded by the pipeline.

```python
from clearml import Task, OutputModel

task = Task.init(project_name="Lednik", task_name="Register teacher")
OutputModel(
    task=task, name="USER-bge-m3", framework="PyTorch",
    config_dict={"hidden_size": 1024},
)
task.close()
```

### 3.2 Student

Initialize a student ([Model Initialization](./model_initialization.md)), save it, and
upload the directory as a model package:

```python
student.save_pretrained("weights/lednik_base")     # writes config.json with `architectures`

om = OutputModel(task=task, name="lednik-base", framework="PyTorch",
                 tags=["Not Distilled"])
om.update_weights_package(weights_path="weights/lednik_base")
```

`config.json` **must** contain `architectures` (written automatically by
`save_pretrained`) — that is how the pipeline resolves the concrete class via
`AutoLednikModel`. The `Not Distilled` tag is dropped automatically from the model
versions the pipeline uploads after distillation.

A student can also be a **Lightning `.ckpt`** from a previous distillation run (upload
the file the same way). Then set in `training_settings.yaml`:

```yaml
is_student_lightning_checkpoint: true
checkpoint_weight_prefix: student      # state-dict prefix to strip
```

### 3.3 Tokenizer

Save the teacher tokenizer to a directory (must contain `tokenizer.json`) and upload it
as a package too; it is loaded back with `AutoTokenizer` under the hood
(`load_tokenizer_from_clearml`).

```python
tokenizer.save_pretrained("weights/tokenizer")
om = OutputModel(task=task, name="user-bge-m3-tokenizer")
om.update_weights_package(weights_path="weights/tokenizer")
```

### 3.4 Datasets

Upload each dataset root from [§2.3](#23-on-disk-layout) as a ClearML Dataset:

```python
from clearml import Dataset

ds = Dataset.create(dataset_project="Lednik", dataset_name="my-dataset")
ds.add_files("my_dataset")           # the folder containing train/ and val/
ds.upload()
ds.finalize()
print(ds.id)                         # goes into training_settings.data.datasets
```

---

## 4. Configuration

The pipeline reads one YAML file (repo default:
[`configs/training_settings.yaml`](../configs/training_settings.yaml)) and connects it
to the task with `connect_as_file`. On a remote run, edits made in the ClearML UI
override the local file. Schemas live in
[`pipelines/distill/configs.py`](../pipelines/distill/configs.py).

```yaml
teacher_model_id: af954495ff03440abc58790c4e41ff22
student_model_id: 316e940db6eb4f95a36f67d9dbc77fe8
tokenizer_id: 34beac693bfa4977b23cafea8b3d6379

model_cfg:                            # how to load the student class
  model_type: lednik                  # "lednik" or "static_embeddings"
  override_params:                    # kwargs forwarded to AutoLednikModel.from_pretrained
    embedding_dropout: 0.15
    attention_dropout: 0.0
    out_attn_dropout: 0.1
    mlp_dropout: 0.1

data:
  datasets:                           # name -> ClearML Dataset ID
    ru-HNP: 79fcccb7143c48f79ddf300bfd5ee1e4
    Mixed-Summarization: 173599873b46449fa9b4ad6350a657bf
  batch_size: 240
  num_workers: 6
  aug_prob: 0.35                      # typo augmentation share (train only)
  query_tok_colname: query-tokens
  query_text_colname: query
  pos_tok_colname: pos-tokens
  pos_text_colname: pos
  query_teacher_embedding_colname: query-embedding
  pos_teacher_embedding_colname: pos-embedding
  #neg_tok_colname: neg-tokens        # hard negatives: all three keys together
  #neg_text_colname: neg
  #neg_teacher_embedding_colname: neg-embedding
  val_label_colname: label            # enables KNN / LogReg metrics
  val_num_labels: 6

trainer:
  accelerator: "cuda"
  max_epochs: 225
  strategy:
    type: "ddp"                       # single_device | ddp | fsdp1 | fsdp2
  val_check_interval: 1.0
  devices: [ 0, 1 ]
  precision: bf16-true
  log_every_n_steps: 25

checkpoint:
  save_top_k: 3
  mode: "min"
  monitor: "val_loss"
  filename: "|{epoch}|-|{step}|-|{val_loss:.5f}|"

#early_stopping:                      # optional
#  monitor: "val_loss"
#  patience: 5
#  mode: "min"
#  min_delta: 0.01

# Online validation: provide `redis`, `runner_config`, or both (see §8).
#redis:
#  host: 127.0.0.1
#  port: 6379
#  stream_name: validation_tasks
runner_config:
  #mrr_config:                        # omit -> MRR skipped
  #  qdrant_host: 127.0.0.1
  #  qdrant_port: 6333
  #  mrr_top_k: 10
  logreg_config:
    lr: 1.0
    weight_decay: 0.01
    solver: "LBFGS"                   # "LBFGS" | "Muon" | "Adam"
    tol: 1e-4
    batch_size: -1
    total_steps: 100
  knn_config:
    k: 5
  scatter_num_points: 200             # points in the PaCMAP scatter plot
  device: "cpu"

distill_config:                       # DistillationConfig, see the no-ClearML guide
  grad_clip_val: 2.0
  distillation_method:
    type: direct-distillation
    distill_loss_type: cosine         # "cosine" or "mse"
    contrastive_loss_weight: 0.7
    temperature: 0.07
  optimizer:
    type: AdamW8bit
    betas: [ 0.9, 0.98 ]
    block_size: 128
    bf16_stochastic_round: true
  lr:
    scheduler_type: plateau-with-cosine-annealing
    warmup_ratio: 0.1
    warmup_value: 1e-5
    base_value: 3e-4
    final_value: 1e-6
    plateau_ratio: 0.8
  weight_decay:
    base_value: 0.02
  freeze_student_emb_steps_ratio: 0.1
```

### `TrainingSettings` fields

| Field                               | Type                               | Description                                                                                                                                           |
| ----------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `teacher_model_id`                | `str`                            | ClearML model ID of the teacher; only`hidden_size` from its config is read.                                                                         |
| `student_model_id`                | `str`                            | ClearML model ID of the student to distill.                                                                                                           |
| `tokenizer_id`                    | `str`                            | ClearML model ID of the tokenizer.                                                                                                                    |
| `model_cfg.model_type`            | `"lednik" \| "static_embeddings"` | Which student family the ID points at.                                                                                                                |
| `model_cfg.override_params`       | `dict`                           | Kwargs applied when loading the student (dropouts etc.).                                                                                              |
| `is_student_lightning_checkpoint` | `bool`                           | `true` if `student_model_id` points to a Lightning `.ckpt`.                                                                                     |
| `checkpoint_weight_prefix`        | `str \| None`                     | Required with the flag above; prefix to strip (usually`student`).                                                                                   |
| `trainer`                         | `LightningTrainerParameters`     | accelerator, devices, strategy, precision,`val_check_interval`, `max_epochs`, `accumulate_grad_batches`, `limit_*_batches`.                   |
| `early_stopping`                  | optional                           | `monitor`, `mode`, `patience`, `min_delta`.                                                                                                   |
| `checkpoint`                      | `CheckpointConfig`               | `save_top_k`, `monitor`, `mode`, `filename`, `save_weights_only`.                                                                           |
| `data`                            | `DataConfig`                     | Datasets, column names,`batch_size`, `num_workers`, `aug_prob`, `val_label_colname`, `val_num_labels`.                                      |
| `redis`                           | `RedisConfig \| None`             | `host`, `port`, `stream_name` — dispatch target for the separate validation worker.                                                            |
| `runner_config`                   | `EvaluationRunnerConfig \| None`  | Local evaluation config.**At least one of `redis` / `runner_config` is required**; with both, Redis is preferred and local is the fallback. |
| `distill_config`                  | `DistillationConfig`             | The objective/optimization schema, documented in[Training without ClearML](./training_without_clearml.md#1-build-the-distillation-config).             |

`trainer.strategy.type` selects the Lightning strategy (`single_device`, `ddp`,
`fsdp1`, `fsdp2`); `setup_strategy` validates the device count (DDP/FSDP need ≥2), and
`DistillationModule.configure_model` performs the actual FSDP wrapping.

---

## 5. How model / checkpoint loading works

The student artifact is fetched from ClearML and loaded by
[`AutoLednikModel`](../lednik/models/auto.py):

```python
clearml_student = InputModel(model_id=settings.student_model_id)
clearml_student.connect(task=task, name="Student Model", ignore_remote_overrides=True)
student_local_path = Path(clearml_student.get_local_copy())

load_kwargs = settings.model_cfg.override_params
if settings.is_student_lightning_checkpoint:
    load_kwargs["weights_prefix"] = settings.checkpoint_weight_prefix
    load_kwargs["strict_prefix"] = True

student = AutoLednikModel.from_pretrained(student_local_path, **load_kwargs)
```

`AutoLednikModel.from_pretrained` resolves the concrete class from the checkpoint
config's `architectures` via the model registry
(see [Model Initialization → Saving and reloading](./model_initialization.md#saving-and-reloading))
and dispatches on the artifact shape:

- **HF package directory** → `model_cls.from_pretrained(local_path, **load_kwargs)`.
- **Lightning `.ckpt`** → the model config is read from the checkpoint's `"config"` key
  (saved by `DistillationModule.on_save_checkpoint`), and `weights_prefix` selects the
  student weights from the training-module state dict. The module stores the student
  under `student.*`, hence `checkpoint_weight_prefix: student` when resuming.

This is why `on_save_checkpoint` both **saves the config dict** and **drops `teacher.*`
keys**: every produced checkpoint is directly reloadable as a bare student model, for
resuming training or for benchmarking.

---

## 6. Checkpoint uploading

During `fit`, checkpoints are written locally to `checkpoints/<task-name>/<task-id>/`
and pushed to ClearML by the `kostyl` `ClearMLLogger`, which wraps an `OutputModel`:

```python
output_model = OutputModel(
    task=task,
    name=clearml_student.name,
    tags=[tag for tag in clearml_student.tags if tag != "Not Distilled"],
    framework="PyTorch",
    comment=f"Model distilled from {clearml_teacher.id}.",
)
logger = ClearMLLogger(
    task=task,
    output_model=output_model,
    upload_checkpoints=True,
    upload_strategy="best",                # upload only improved checkpoints
    model_config_provider=lambda: distillation_module.model_config,
)
```

Alongside the logger, the pipeline wires a `LearningRateMonitor`, the checkpoint
callback and, if configured, `EarlyStopping` — all `kostyl` helpers.

---

## 7. Running the pipeline

```bash
# Local run (uses trainer.devices)
uv run python -m pipelines.distill.run --config-path configs/training_settings.yaml

# Remote run on a ClearML agent queue
uv run python -m pipelines.distill.run --config-path configs/training_settings.yaml \
    --remote-execution-queue my-queue

# Tags and task reuse
uv run python -m pipelines.distill.run --config-path configs/training_settings.yaml \
    --tags "exp1,baseline" --no-reuse-last-task-id
```

| Flag                                               | Default  | Description                                                                                                  |
| -------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------ |
| `--config-path <file>`                           | required | The`TrainingSettings` YAML.                                                                                |
| `--remote-execution-queue <name>`                | `""`   | If non-empty,`task.execute_remotely(queue)` is called and the local process exits; an agent runs the task. |
| `--tags tag1,tag2`                               | `""`   | Comma-separated tags added to the ClearML task.                                                              |
| `--reuse-last-task-id / --no-reuse-last-task-id` | `True` | Reuse the previous task ID or always create a new one.                                                       |

The task is created as `project_name="Lednik"`, `task_name="Model Distillation"`,
`task_type=training`; the run is seeded with `seed_everything(42)`. PyTorch
auto-logging is disabled; TensorBoard and matplotlib capture are on.

---

## 8. Online validation

At the end of **every validation epoch**, `DistillationModule` packs the val embeddings
into a `ValidationContract` and hands it to the `EvaluationDispatcher`
([`lednik/distill/validation/`](../lednik/distill/validation)). How often validation
epochs happen is a Lightning trainer setting (`trainer.val_check_interval`): a
fractional value runs validation several times per training epoch (e.g. `0.3` → three
validation epochs per training epoch), `1.0` — once at its end. Per contract, the
metrics — KNN accuracy, logistic-regression probe, MRR (retrieval, backed by Qdrant),
plus a PaCMAP scatter — are computed and logged as scalars to the originating ClearML
task.

Where they are computed depends on the config:

| `redis` | `runner_config` | Behavior                                                                                                                 |
| --------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------ |
| set       | —                | Contracts go to a Redis stream; a**separate worker** consumes them. Training never blocks on metrics.              |
| —        | set               | No infrastructure: the metrics run**locally, in-process**, after each validation epoch. The repo default.          |
| set       | set               | Redis is used while reachable; on dispatch failure the pipeline **falls back to local** evaluation and continues. |

At least one of the two must be configured — the module refuses to start otherwise.

### Running the separate worker

The worker is an independent process; run it on the training host or on a different
machine — it only needs Redis (and Qdrant for MRR) to be reachable.

```bash
# infrastructure (docker-compose.yaml, `training` profile)
docker compose --profile training up -d redis     # dispatch
docker compose --profile training up -d qdrant    # only needed for MRR

uv run python -m lednik.distill.validation.worker --config-path configs/worker_config.yaml
```

[`configs/worker_config.yaml`](../configs/worker_config.yaml) is an
`EvaluationWorkerConfig`: the same `runner_config` schema as in
`training_settings.yaml`, plus the Redis connection:

```yaml
redis:
  host: localhost
  port: 6379
  stream_name: validation_tasks      # must match training_settings.redis.stream_name
runner_config:
  mrr_config:
    qdrant_host: localhost
    qdrant_port: 6333
    mrr_top_k: 10
  logreg_config:
    lr: 1.0
    weight_decay: 0.01
    solver: "LBFGS"
    tol: 1e-4
    batch_size: -1
    total_steps: 100
  knn_config:
    k: 5
  scatter_num_points: 200
  device: cpu
```

The worker creates the consumer group, blocks on the stream, evaluates each contract,
acknowledges it, and logs the results to the task that produced it. Stop with `Ctrl-C`.

Notes on the metrics themselves:

- `mrr_config` / `logreg_config` / `knn_config` are each optional — omit one to skip
  that metric. `scatter_num_points` controls the PaCMAP plot size.
- KNN and LogReg need labels: set `data.val_label_colname` and `data.val_num_labels`.
- MRR needs a reachable Qdrant.

---

## 9. Benchmarking with MTEB

[`bench/mteb_testing/run.py`](../bench/mteb_testing/run.py) evaluates a model on
Russian MTEB tasks through the
[`MTEBModelWrapper`](../bench/mteb_testing/model_wrapper.py) adapter, loading either a
ClearML model or a HuggingFace model:

```bash
uv sync --group bench

# a ClearML-registered Lednik/Static model
uv run python -m bench.mteb_testing.run --clearml \
    --model-id <CLEARML_MODEL_ID> \
    --tokenizer-id <CLEARML_TOKENIZER_ID> \
    --output-file bench/mteb_testing/results/metrics.jsonl \
    --batch-size 128

# a plain HuggingFace model
uv run python -m bench.mteb_testing.run --no-clearml \
    --model-id deepvk/USER-bge-m3 \
    --output-file bench/mteb_testing/results/metrics.jsonl
```

For ClearML models the runner reads `model_type` from the model config to pick the
class, and passes `weights_prefix="student"` automatically when the model is tagged
`LightningCheckpoint`. Results append to the given `.jsonl`, one line per run.
