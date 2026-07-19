# Training with ClearML

This guide covers the **production distillation pipeline** in
[`pipelines/distill/`](../pipelines/distill). It builds on the same
`DistillationModule`/`ContrastiveCollator` described in
[Training without ClearML](./training_without_clearml.md), and adds everything
[ClearML](https://clear.ml/) gives you:

- **Config syncing** — YAML configs are connected to the ClearML task and can be overridden
  from the UI for remote runs.
- **Artifact loading** — the teacher, student and tokenizer are pulled from the ClearML
  model registry by ID; datasets are pulled from ClearML Datasets.
- **Checkpoint uploading** — best/every checkpoints are pushed back to the registry as new
  model versions.
- **Remote execution** — enqueue the run on a ClearML agent queue with one flag.
- **Online validation** — embeddings are streamed to a separate worker that computes
  KNN / LogReg / MRR metrics against Redis + Qdrant.

Almost all ClearML/Lightning glue comes from the
[`kostyl`](./model_initialization.md#about-kostyl) submodule
(`kostyl.ml.integrations.clearml`, `kostyl.ml.integrations.lightning`).

---

## Pipeline at a glance

```
                 configs/training_settings.yaml
                     (incl. distill_config section)
                              │  connect_as_file (ClearML config sync)
                              ▼
   ClearML registry ──► load teacher (InputModel: hidden_size only)
        │              load student (load_model_from_clearml)
        │              load tokenizer (load_tokenizer_from_clearml)
        ▼
   ClearML Datasets ─► DataModule (download + load train/ val/ splits)
        │                 │  ContrastiveCollator
        ▼                 ▼
   DistillationModule (Lightning) ──fit──► checkpoints ──► ClearMLCheckpointUploader
        │                                                       (new model versions)
        └── validation embeddings ──Redis stream──► EvaluationWorker (KNN/LogReg/MRR)
                                                          └── Qdrant + ClearML scalars
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
clearml-init
```

For S3-backed artifact storage, the `clearml[s3]` extra is already pulled in by the
`distill` group; configure your bucket in `clearml.conf`.

---

## 2. Prepare ClearML artifacts

The pipeline references three models and N datasets **by ClearML ID**.

### 2.1 Teacher, student, tokenizer (model registry)

- **Teacher** (`teacher_model_id`) — only its `config_dict["hidden_size"]` is read (to size
  the student→teacher projection). Register the teacher transformer as a ClearML model with
  a config that includes `hidden_size`.
- **Student** (`student_model_id`) — the initialized student from
  [Model Initialization](./model_initialization.md). Save with `save_pretrained(...)` and
  register the directory as a ClearML model. Its config **must** contain `architectures`
  (written automatically by `save_pretrained`) so the pipeline can resolve the class via
  `AutoLednikModel` and the model registry.
- **Tokenizer** (`tokenizer_id`) — the teacher tokenizer, registered as a model;
  loaded with `AutoTokenizer.from_pretrained` under the hood.

A student can be registered in two shapes, both handled by `load_model_from_clearml`:

| Shape | How it was produced | `training_settings` flags |
| --- | --- | --- |
| **HF package directory** | `model.save_pretrained(dir)` | `is_student_lightning_checkpoint: false` (default) |
| **Lightning `.ckpt`** | A checkpoint from a previous distillation run | `is_student_lightning_checkpoint: true` + `checkpoint_weight_prefix: <prefix>` |

See [How model / checkpoint loading works](#how-model--checkpoint-loading-works) for the
mechanism.

### 2.2 Datasets (ClearML Datasets)

`training_settings.data.datasets` maps a human-readable name → ClearML **Dataset ID**. Each
dataset, once downloaded, must contain `train/` and `val/` subfolders, where each subfolder
is a `datasets.Dataset` saved with `save_to_disk`. The `DataModule`:

1. `download_clearml_datasets(...)` in `prepare_data` (parallel download).
2. globs `**/train/` and `**/val/` inside each dataset
   (`_collect_subset_paths`).
3. `load_from_disk` each split, drops all columns except the configured ones, and
   `concatenate_datasets`.

The columns each split must contain are exactly the
[raw-row columns from the collator contract](./training_without_clearml.md#21-raw-dataset-rows-input-to-the-collator):
`query_tok_colname`, `pos_tok_colname`, `query_teacher_embedding_colname`,
`pos_teacher_embedding_colname` (+ negatives if configured), and — for `val/` only —
`val_label_colname` if set.

Expected on-disk layout per dataset:

```
<dataset_root>/
├── train/        # datasets.save_to_disk output
│   └── ... (dataset_info.json, state.json, *.arrow)
└── val/
    └── ...
```

---

## 3. Configuration files

The pipeline reads [`configs/training_settings.yaml`](../configs/training_settings.yaml)
and connects it to the ClearML task with `connect_as_file` (from
`ConfigSyncingClearmlMixin`). On a remote run, edits made in the ClearML UI override the
local file. The distillation hyperparameters live in the `distill_config` **section** of
that same file.

### 3.1 The `distill_config` section

This is a [`DistillationConfig`](../pipelines/distill/configs.py) — the same schema as the
[no-ClearML config](./training_without_clearml.md#1-build-the-distillation-config). Example:

```yaml
grad_clip_val: 2.0
teacher_pooling_method: last
distillation_method:
  type: direct-distillation
  distill_loss_type: cosine          # "cosine" or "mse"
  contrastive_loss_weight: 1.0       # 1.0 -> pure contrastive (distill loss disabled)
  temperature: 0.07
optimizer:
  type: AdamW8bit
  betas: [0.9, 0.98]
  block_size: 128
  bf16_stochastic_round: true
lr:
  scheduler_type: plateau-with-cosine-annealing
  warmup_ratio: 0.2
  warmup_value: 8e-5
  base_value: 3e-4
  final_value: 6e-5
  plateau_ratio: 0.7
weight_decay:
  base_value: 0.001
freeze_student_emb_steps_ratio: 0.1
```

### 3.2 The rest of `configs/training_settings.yaml`

The top level is a [`TrainingSettings`](../pipelines/distill/configs.py) model — it ties
together artifact IDs, the student build config, trainer/strategy settings, checkpointing,
data and (optional) Redis. Example (the repo default):

```yaml
redis:                               # optional; enables online validation dispatch
  host: 127.0.0.1
  port: 6379
  stream_name: validation_tasks

data:
  datasets:
    ru-HNP: 79fcccb7143c48f79ddf300bfd5ee1e4    # name: clearml_dataset_id
    Mixed-Summarization: 173599873b46449fa9b4ad6350a657bf
  batch_size: 96
  num_workers: 4
  query_tok_colname: query-tokens
  pos_tok_colname: pos-tokens
  #neg_tok_colname: neg-tokens
  query_teacher_embedding_colname: query-embedding
  pos_teacher_embedding_colname: pos-embedding
  #neg_teacher_embedding_colname: neg-embedding
  val_label_colname: label            # used for KNN validation metrics
  val_num_labels: 6                   # number of classes -> DistillationModule(num_labels=...)

trainer:
  accelerator: "cuda"
  max_epochs: 20
  strategy:
    type: "ddp"                       # single_device | ddp | fsdp1 | fsdp2
  val_check_interval: 0.3
  devices: 8
  precision: bf16-true
  log_every_n_steps: 25

early_stopping:                       # optional
  monitor: "val_loss"
  patience: 5
  mode: "min"
  min_delta: 0.01

checkpoint:
  save_top_k: 3
  mode: "min"
  monitor: "val_loss"
  filename: "|{epoch}|-|{step}|-|{val_loss:.5f}|"

teacher_model_id: af954495ff03440abc58790c4e41ff22
student_model_id: 2064bdd9dda642e0bcf05ef10cc7595a
tokenizer_id: 34beac693bfa4977b23cafea8b3d6379

model_cfg:                            # how to (re)build the student class on load
  model_type: lednik                  # "lednik" or "static_embeddings"
  embedding_dropout: 0.25
  attention_dropout: 0.0
  out_attn_dropout: 0.15
  mlp_dropout: 0.15

# resume from a Lightning checkpoint instead of an HF package:
# is_student_lightning_checkpoint: true
# checkpoint_weight_prefix: student
```

#### `TrainingSettings` fields

| Field | Type | Description |
| --- | --- | --- |
| `teacher_model_id` | `str` | ClearML model ID of the teacher (only `hidden_size` is read). |
| `student_model_id` | `str` | ClearML model ID of the student to distill. |
| `tokenizer_id` | `str` | ClearML model ID of the tokenizer. |
| `model_cfg` | `LednikModelTrainConfig \| StaticEmbeddingsTrainConfig` | `model_type` + dropout overrides applied when the student is loaded. |
| `is_student_lightning_checkpoint` | `bool` | `true` if `student_model_id` points to a Lightning `.ckpt`. |
| `checkpoint_weight_prefix` | `str \| None` | Required when the above is `true`; the state-dict prefix to strip (e.g. `student`). |
| `trainer` | `LightningTrainerParameters` | accelerator, devices, strategy, precision, `val_check_interval`, `max_epochs`, `log_every_n_steps`, `accumulate_grad_batches`, `limit_*_batches`. |
| `early_stopping` | `EarlyStoppingConfig \| None` | `monitor`, `mode`, `patience`, `min_delta`. |
| `checkpoint` | `CheckpointConfig` | `save_top_k`, `monitor`, `mode`, `filename`, `save_weights_only`. |
| `data` | `DataConfig` | Datasets + column names + batch/worker settings (see §2.2). |
| `redis` | `RedisConfig \| None` | Online-validation dispatch target. Omit to disable. |

`trainer.strategy.type` selects the strategy: `single_device`, `ddp`, `fsdp1`, `fsdp2`.
`setup_strategy` validates the device count (FSDP/DDP need ≥2 devices) and builds the
matching Lightning strategy; `DistillationModule.configure_model` performs the actual FSDP
wrapping.

---

## How model / checkpoint loading works

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

`AutoLednikModel.from_pretrained` resolves the concrete class from the checkpoint config's
`architectures` via the model registry
(see [Model Initialization → Saving and reloading](./model_initialization.md#saving-and-reloading))
and dispatches on the artifact shape:

- **HF package directory** → `model_cls.from_pretrained(local_path, **load_kwargs)`. The
  `model_cfg` overrides are applied here (`weights_prefix`/`strict_prefix` are dropped).
- **Lightning `.ckpt`** → the model config is read from the checkpoint's `"config"` key
  (saved by `DistillationModule.on_save_checkpoint`), and `weights_prefix` selects the
  student weights from the training-module state dict. The `DistillationModule` stores the
  student under `student.*`, so when resuming from a distillation checkpoint set
  `checkpoint_weight_prefix: student`.

This is why `on_save_checkpoint` both **saves the config dict** and **drops `teacher.*`
keys**: it makes every produced checkpoint directly reloadable as a bare student model, for
resuming training or for benchmarking.

The tokenizer is loaded with `load_tokenizer_from_clearml(model_id=tokenizer_id, ...)`.

---

## Checkpoint uploading

During `fit`, checkpoints are written locally (to `checkpoints/<task-name>/<task-id>/`, per
the `checkpoint:` section of `training_settings.yaml`) and pushed back to ClearML through
the `kostyl` `ClearMLLogger`, which wraps an `OutputModel`:

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

Alongside the logger, the pipeline wires a `LearningRateMonitor`, the checkpoint callback
(`setup_checkpoint_callback`) and, if `early_stopping` is configured, an `EarlyStopping`
callback (`setup_early_stopping_callback`) — all `kostyl` helpers.

---

## Running the pipeline

```bash
# Local run (uses the GPUs in trainer.devices)
python -m pipelines.distill.run

# Remote run on a ClearML agent queue
python -m pipelines.distill.run --remote-execution-queue my-queue

# Add tags and control task reuse
python -m pipelines.distill.run --tags "exp1,baseline" --no-reuse-last-task-id
```

CLI options (see [`run.py`](../pipelines/distill/run.py)):

| Flag | Default | Description |
| --- | --- | --- |
| `--remote-execution-queue <name>` | `""` | If non-empty, `task.execute_remotely(queue)` is called and the local process exits; an agent runs the task. |
| `--tags tag1,tag2` | `""` | Comma-separated tags added to the ClearML task. |
| `--reuse-last-task-id / --no-reuse-last-task-id` | `True` | Reuse the previous task ID (useful for iterating) or always create a new task. |

The task is created as `project_name="Lednik"`, `task_name="Model Distillation"`,
`task_type=training`. PyTorch auto-logging is disabled; TensorBoard + matplotlib capture is
on.

---

## Online validation worker

When `training_settings.redis` is set, `DistillationModule` streams validation embeddings to
a Redis stream at the end of each validation epoch (via `EvaluationDispatcher`). A separate
long-running **worker** consumes the stream and computes embedding-quality metrics, logging
scalars back to the originating ClearML task.

Metrics implemented ([`lednik/distill/validation/metrics/`](../lednik/distill/validation/metrics)):
KNN accuracy, logistic-regression probing, and MRR (retrieval, backed by Qdrant).

### 1. Start infrastructure

[`docker-compose.yaml`](../docker-compose.yaml) provides Redis and Qdrant:

```bash
docker compose up redis             # required for dispatch
docker compose --profile qdrant up qdrant   # required only for MRR
```

### 2. Configure the worker

[`configs/worker_config.yaml`](../configs/worker_config.yaml) is an
`EvaluationWorkerConfig` ([`lednik/distill/validation/structs/configs.py`](../lednik/distill/validation/structs/configs.py)):

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
    solver: "LBFGS"                   # "LBFGS" | "Muon" | "Adam"
    tol: 1e-4
    batch_size: -1
    total_steps: 100
  knn_config:
    k: 5
  scatter_num_points: 200
  device: cpu
```

Any of `mrr_config` / `logreg_config` / `knn_config` may be omitted to skip that metric.
`val_num_labels` in `training_settings` is what becomes `num_labels` for the KNN metric.

### 3. Run the worker

```bash
python -m lednik.distill.validation.worker --config-path configs/worker_config.yaml
```

The worker creates the consumer group, blocks on the stream, evaluates each
`ValidationContract`, acknowledges it, and logs results to ClearML and Qdrant. Stop it with
`Ctrl-C`.

---

## Benchmarking with MTEB

[`bench/mteb_testing/run.py`](../bench/mteb_testing/run.py) evaluates a model on Russian MTEB tasks, using
the [`MTEBModelWrapper`](../bench/mteb_testing/model_wrapper.py) adapter. It can load either a
ClearML model or a HuggingFace model.

```bash
# Evaluate a ClearML-registered Lednik/Static model
python -m bench.mteb.run --clearml \
    --model-id <CLEARML_MODEL_ID> \
    --tokenizer-id <CLEARML_TOKENIZER_ID> \
    --output-file results/mteb.jsonl \
    --pooling mean --batch-size 128

# Evaluate a plain HuggingFace model
python -m bench.mteb.run --no-clearml \
    --model-id deepvk/USER-bge-m3 \
    --output-file results/mteb.jsonl
```

For ClearML models the runner reads `model_type` from the model config to pick the class,
and auto-passes `weights_prefix="student"` when the model is tagged `LightningCheckpoint`.
Install the bench extras first:

```bash
uv sync --group bench
```

The dependency group is `bench = ["mteb[faiss-cpu]"]`, declared in
[`pyproject.toml`](../pyproject.toml).
