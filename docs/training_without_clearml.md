# Training without ClearML

This guide shows how to run distillation **purely from Python**, without ClearML,
remote queues, or config syncing. You assemble the pieces yourself:

1. Build a [`DistillationConfig`](#1-build-the-distillation-config).
2. Understand the [data format the collator produces](#2-data-format-the-collator-contract)
   so the training module can consume it.
3. Instantiate the [`DistillationModule`](#3-instantiate-the-training-module).
4. Wire a `DataLoader` + a `lightning.Trainer` and call `fit`.

This is the right path for notebooks, unit tests, custom data sources, or any environment
where you don't want a ClearML server. If you *do* want artifact management, config
versioning and online validation, read [Training with ClearML](./training_with_clearml.md)
instead — it wraps everything below.

> All training logic lives in
> [`lednik/distill/training_module.py`](../lednik/distill/training_module.py) and
> [`lednik/distill/collator.py`](../lednik/distill/collator.py). The module subclasses
> `KostylLightningModule` from the [`kostyl`](./model_initialization.md#about-kostyl)
> submodule, which provides gradient clipping, checkpoint-config saving and scheduled-value
> logging.

---

## 0. Prerequisites

```python
import torch
from transformers import AutoTokenizer
from lednik.models import LednikModel  # or StaticEmbeddingsModel
```

You need:

- A **student** model — initialized as in [Model Initialization](./model_initialization.md).
- The **teacher's tokenizer** (the student was seeded from it).
- The **teacher hidden size** — the dimensionality of the precomputed teacher sentence
  embeddings (e.g. `teacher.config.hidden_size`). The module builds a
  `student → teacher` projection from this.
- A dataset where **teacher sentence embeddings are precomputed offline**. Lednik does
  *not* run the teacher during training; it only trains the student against stored teacher
  vectors. (Precomputing them is also how the ClearML pipeline works.)

---

## 1. Build the distillation config

[`DistillationConfig`](../lednik/distill/configs.py) is a `pydantic` model (it extends
`kostyl.ml.configs.HyperparamsConfig`). Build it directly in Python, or validate a dict
loaded from YAML/JSON.

```python
from lednik.distill.configs import DistillationConfig

train_cfg = DistillationConfig.model_validate(
    {
        # ---- distillation objective ----
        "distillation_method": {
            "type": "direct-distillation",
            "distill_loss_type": "cosine",      # "cosine" or "mse"
            "contrastive_loss_weight": 0.7,     # 0..1, see "How the loss works" below
            "temperature": 0.07,                # required when contrastive_loss_weight > 0
        },

        # ---- optimization (schemas come from kostyl) ----
        "optimizer": {
            "type": "AdamW8bit",                # AdamW | Adam | Adam8bit | AdamW8bit | Muon | ...
            "betas": [0.9, 0.98],
            "block_size": 128,                  # required for *8bit/*4bit/*Fp8 optimizers
            "bf16_stochastic_round": True,
        },
        "lr": {
            "scheduler_type": "plateau-with-cosine-annealing",
            "warmup_ratio": 0.2,
            "warmup_value": 8e-5,
            "base_value": 3e-4,
            "final_value": 6e-5,
            "plateau_ratio": 0.7,
        },
        "weight_decay": {"base_value": 1e-3},

        # ---- optional knobs ----
        "grad_clip_val": 2.0,
        "freeze_student_emb_steps_ratio": 0.1,  # freeze student embeddings for first 10% of steps
        "embeddings_lr_multiplier": None,
    }
)
```

### Field reference

| Field                                           | Type                         | Required    | Description                                                                       |
| ----------------------------------------------- | ---------------------------- | ----------- | --------------------------------------------------------------------------------- |
| `distillation_method`                         | `DirectDistillationConfig` | ✓          | The objective (see below).                                                        |
| `distillation_method.type`                    | `"direct-distillation"`    | ✓          | Only method currently available.                                                  |
| `distillation_method.distill_loss_type`       | `"cosine" \| "mse"`         | —          | Token/sentence regression loss (default`"cosine"`).                             |
| `distillation_method.contrastive_loss_weight` | `float` 0–1               | —          | Mix between contrastive and regression loss (default`0.7`).                     |
| `distillation_method.temperature`             | `float > 0`                | conditional | **Required** when `contrastive_loss_weight > 0`.                          |
| `optimizer`                                   | optimizer config             | ✓          | `kostyl` optimizer schema; `type` selects Adam/AdamW/8-bit/Muon variants.     |
| `lr`                                          | `Lr`                       | ✓          | Learning-rate schedule (see[scheduling](#learning-rate--weight-decay-scheduling)). |
| `weight_decay`                                | `WeightDecay`              | ✓          | Same schedule schema as`lr`; usually just `base_value`.                       |
| `grad_clip_val`                               | `float > 0 \| None`         | —          | Gradient-norm clip (applied by the Lightning module, not the Trainer).            |
| `freeze_student_emb_steps_ratio`              | `float` 0–1 \| `None`   | —          | Fraction of total steps to keep the student embedding table frozen.               |
| `embeddings_lr_multiplier`                    | `float > 0 \| None`         | —          | Separate LR multiplier for the embedding table.                                   |

### Learning-rate / weight-decay scheduling

`lr` and `weight_decay` use the `kostyl` scheduled-parameter schema. `scheduler_type` is
one of:

- `"linear"`
- `"cosine"`
- `"plateau-with-cosine-annealing"`
- `"plateau-with-linear-annealing"`
- `None` (constant)

Relevant fields: `base_value` (required), `warmup_value`, `warmup_ratio`, `final_value`,
`plateau_ratio`, `freeze_ratio`. The module turns these into schedulers via
`kostyl.ml.optim.create_scheduler` inside `configure_optimizers`, and — when running
distributed — scales learning rates by world size with
`kostyl.ml.dist_utils.scale_lrs_by_world_size`.

---

## How the loss works

`DistillationModule._base_step` computes a weighted sum of two losses:

```
loss = contrastive_loss_weight       * contrastive_loss
     + (1 - contrastive_loss_weight) * distill_loss
```

- **`distill_loss`** (regression): the student's sentence embedding is projected to the
  teacher dimension (`student_to_teacher_proj`, a `Linear`, or `Identity` when dims match)
  and compared to the stored teacher embedding via cosine or MSE. Skipped entirely when
  `contrastive_loss_weight == 1.0`.
- **`contrastive_loss`** (InfoNCE): a symmetric in-batch contrastive loss between query and
  positive sentence embeddings (plus optional hard negatives), gathered across ranks when
  distributed. Skipped when `contrastive_loss_weight == 0.0`. Requires `temperature`.

Both losses operate on **sentence embeddings** — there is no token-level term. Special
tokens participate normally for a `LednikModel` student; for a `StaticEmbeddingsModel`
they are masked out of mean pooling (a lookup table has no contextual vector for
`[CLS]`/`[SEP]`, they would only add noise).

---

## 2. Data format: the collator contract

This is the contract you must satisfy. There are two layers: the **raw dataset rows** you
provide, and the **batched tensors** the collator emits (which `_base_step` consumes).

### 2.1 Raw dataset rows (input to the collator)

[`ContrastiveCollator`](../lednik/distill/collator.py) reads each row by **column name**.
Column names are configurable via the collator's constructor arguments. Every logical
field is a **triple**: token ids, the raw text, and the precomputed teacher embedding.
A row is a `dict` with:

| Logical field                                    | Constructor arg                                                              | Type                                               | Notes                                                                                                      |
| ------------------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Query tokens                                     | `query_tok_colname`                                                        | `list[int]` **or** `list[list[int]]`     | Token ids**without** special tokens. If a list of lists, one variant is sampled at random per epoch. |
| Query text                                       | `query_text_colname`                                                       | `str` **or** `list[str]`                 | Raw text of the same query; used for typo augmentation (re-tokenized on the fly).                          |
| Query teacher embedding                          | `query_teacher_embedding_colname`                                          | `list[float]` **or** `list[list[float]]` | Precomputed teacher sentence vector of the**clean** text.                                            |
| Positive tokens / text / embedding               | `pos_tok_colname`, `pos_text_colname`, `pos_teacher_embedding_colname` | same as query                                      | —                                                                                                         |
| Negative tokens / text / embedding*(optional)* | `neg_tok_colname`, `neg_text_colname`, `neg_teacher_embedding_colname` | same as query                                      | Hard negatives. All three must be provided**together** or omitted together.                          |
| Label*(optional)*                              | `label_colname`                                                            | `int` or `list[int]`                           | Only used for KNN-style validation metrics; ignored if not set.                                            |

Important details:

- **Token ids carry no special tokens.** The collator's post-processor adds them:
  `[CLS] … [SEP]` if the tokenizer has `cls`/`sep`, otherwise `… [EOS]`. Sequences are
  truncated to `max_len` (defaults to `tokenizer.model_max_length`).
- **Variants.** If a field holds multiple variants (paraphrases/crops), tokens, text and
  embedding must have the **same variant count**; the collator samples one index per row
  and uses it for all three.
- **Teacher embedding dimensionality** must equal the `teacher_hidden_size` you pass to the
  module.

A minimal raw row (single variant, teacher dim 1024):

```python
row = {
    "query":            "как оформить возврат товара",
    "query-tokens":     [4211, 8765, 1289, 6710],   # no [CLS]/[SEP]; collator adds them
    "query-embedding":  [0.012, -0.4, ...],         # length == teacher_hidden_size
    "pos":              "процедура возврата покупки",
    "pos-tokens":       [5012, 990, 15672],
    "pos-embedding":    [0.10, 0.02, ...],
    # optional:
    # "neg":            "...",
    # "neg-tokens":     [...],
    # "neg-embedding":  [...],
    # "label":          3,
}
```

**Typo augmentation.** With `aug_prob > 0` (and a `corruptor` provided —
[SAGE](https://github.com/ai-forever/sage) `SBSCCorruptor`), that fraction of rows per
batch gets its *text* corrupted with realistic typos and re-tokenized; the teacher
embedding stays the clean-text vector, so the student learns to map noisy input onto
clean embeddings. Texts that fail corruption `max_aug_attempts` times are blacklisted
and used verbatim. Keep `aug_prob=0.0` for validation collators.

### 2.2 Batched output (`CollatorOutput`)

Calling the collator on a list of rows yields a `CollatorOutput` `TypedDict`. Rows are
stacked **in blocks**: all queries first, then all positives, then all negatives (if used).
Teacher embeddings are concatenated in the **same order**, so row `i` of `input_ids`
corresponds to row `i` of `teacher_sentence_embeddings`.

For a batch of `B` rows (so `B` queries + `B` positives, `Bn` negatives, padded length `L`):

| Key                             | Shape                        | dtype    | Meaning                                                                    |
| ------------------------------- | ---------------------------- | -------- | -------------------------------------------------------------------------- |
| `input_ids`                   | `(N, L)`                   | `long` | Padded token ids.`N = B + B (+ Bn)`. `L` is padded to a multiple of 8. |
| `attention_mask`              | `(N, L)`                   | `long` | `1` for real tokens, `0` for padding.                                  |
| `queries_mask`                | `(N,)`                     | `bool` | `True` on the query rows (first block).                                  |
| `positives_mask`              | `(N,)`                     | `bool` | `True` on the positive rows (second block).                              |
| `negatives_mask`              | `(N,)` or `None`         | `bool` | `True` on the negative rows, or `None` when no negatives.              |
| `labels`                      | `(B,)`                     | `long` | Per-query labels, or a`-1` fill tensor when no labels.                   |
| `teacher_sentence_embeddings` | `(N, teacher_hidden_size)` | float    | Teacher vectors aligned row-for-row with`input_ids`.                     |

`drop_last=True` plus equal-count query/positive blocks is assumed by the contrastive loss
(it builds an identity target `arange` aligning query `i` to positive `i`).

### 2.3 Constructing the collator

```python
from lednik.distill.collator import ContrastiveCollator

collator = ContrastiveCollator(
    tokenizer=tokenizer,
    query_tok_colname="query-tokens",
    query_text_colname="query",
    query_teacher_embedding_colname="query-embedding",
    pos_tok_colname="pos-tokens",
    pos_text_colname="pos",
    pos_teacher_embedding_colname="pos-embedding",
    # neg_tok_colname="neg-tokens",                       # optional, all three together
    # neg_text_colname="neg",
    # neg_teacher_embedding_colname="neg-embedding",
    label_colname=None,                                   # set to enable KNN-val labels
    pad_to_multiple_of=8,
    max_len=None,                                         # None -> tokenizer.model_max_length
    aug_prob=0.0,                                         # >0 needs a corruptor:
    # corruptor=SBSCCorruptor.from_config(SBSCConfig(min_typos=5)),  # from sage
)
```

---

## 3. Instantiate the training module

```python
from lednik.distill.training_module import DistillationModule
from lednik.distill.validation import EvaluationRunnerConfig
from kostyl.ml.configs.training_settings import SingleDeviceStrategyConfig

module = DistillationModule(
    student=student,                       # StaticEmbeddingsModel | LednikModel
    tokenizer=tokenizer,
    teacher_hidden_size=1024,              # dim of stored teacher embeddings
    train_cfg=train_cfg,                   # the DistillationConfig from step 1
    strategy_config=SingleDeviceStrategyConfig(type="single_device"),
    task=None,                             # online validation is ClearML-only -> losses only
    runner_config=EvaluationRunnerConfig(  # required by the module, unused without a task
        scatter_num_points=200,
    ),
    redis_config=None,
    num_labels=None,
)
```

**Online validation (KNN / LogReg / MRR) is not supported in this path** — it is a
ClearML feature: the dispatched `ValidationContract` carries a task id, and the metric
scalars are logged into that ClearML task. With `task=None` the module skips dispatch
entirely, so validation logs only the losses (`loss`, `CosineSimilarity`, `RMSE`). The
`runner_config` / `redis_config` parameters still have to satisfy the module's
validator — at least one must be provided; a minimal
`EvaluationRunnerConfig(scatter_num_points=200)` does the job and stays unused. For the
online metrics, use the ClearML pipeline — see
[Training with ClearML → Online validation](./training_with_clearml.md#8-online-validation).

### `DistillationModule` parameters

| Parameter               | Type                                    | Description                                                                                                                                                                                            |
| ----------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `student`             | `StaticEmbeddingsModel \| LednikModel` | Model to train.                                                                                                                                                                                        |
| `tokenizer`           | tokenizer                               | Used to find special-token ids (masked out of pooling for a static student).                                                                                                                           |
| `teacher_hidden_size` | `int`                                 | Teacher embedding dim; sizes the`student → teacher` projection.                                                                                                                                     |
| `train_cfg`           | `DistillationConfig`                  | Hyperparameters/objective.                                                                                                                                                                             |
| `strategy_config`     | `SUPPORTED_STRATEGIES`                | `kostyl` strategy config: `SingleDeviceStrategyConfig`, `DDPStrategyConfig`, `FSDP1StrategyConfig`, or `FSDP2StrategyConfig`. Must match the Lightning strategy you pass to the `Trainer`. |
| `task`                | `clearml.Task \| None`                 | Online metrics run only when set (scalars are logged to this task); `None` → dispatch is skipped, validation logs only losses.                                                                        |
| `runner_config`       | `EvaluationRunnerConfig \| None`       | Local in-process evaluation: optional`knn_config` / `logreg_config` / `mrr_config`, `scatter_num_points`, `device`.                                                                          |
| `redis_config`        | `RedisConfig \| None`                  | Dispatch target (`host`, `port`, `stream_name`) for a separate validation worker. At least one of `runner_config` / `redis_config` is required.                                              |
| `num_labels`          | `int \| None`                          | Number of classes for the KNN / LogReg validation metrics.                                                                                                                                             |

> The class imports `from clearml import Task` at module load, so the `clearml` package
> must be installed (it ships in the `distill` dependency group) even when you pass
> `task=None`. You simply never call `Task.init`, so no server is contacted.

**Strategy ↔ Trainer must agree.** `SingleDeviceStrategyConfig` ⇒ `strategy="auto"` on one
device. `DDPStrategyConfig`/`FSDP1StrategyConfig`/`FSDP2StrategyConfig` require ≥2 devices
and the matching Lightning strategy. The module's `configure_model` does the FSDP wrapping
itself based on `strategy_config`.

---

## 4. Train

```python
import lightning as L
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")

train_loader = DataLoader(
    train_dataset,                 # any Dataset of raw rows (e.g. a HF datasets.Dataset)
    batch_size=64,
    shuffle=True,
    drop_last=True,                # required: keeps query/positive blocks equal-sized
    num_workers=4,
    pin_memory=True,
    collate_fn=collator,           # the ContrastiveCollator from step 2.3
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=True,
    num_workers=4,
    collate_fn=collator,
)

trainer = L.Trainer(
    max_epochs=10,
    accelerator="cuda",            # or "cpu"
    devices=1,
    strategy="auto",               # must match strategy_config above
    precision="bf16-true",
    gradient_clip_val=None,        # leave None — the module clips via train_cfg.grad_clip_val
    val_check_interval=0.3,
)

trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

Notes:

- Keep `gradient_clip_val=None` on the `Trainer`. Clipping is driven by
  `train_cfg.grad_clip_val` inside the `KostylLightningModule`.
- Online KNN / LogReg / MRR validation is ClearML-only (see the note in §3); in this
  path validation logs `loss`, `CosineSimilarity` and `RMSE`.
- `on_save_checkpoint` stores the model `config` dict in the checkpoint and strips any
  `teacher.*` keys, so the resulting `.ckpt` can later be loaded into a bare model via
  `Model.from_lightning_checkpoint(...)` (see
  [Training with ClearML → checkpoint loading](./training_with_clearml.md#how-model--checkpoint-loading-works)).

---

## Putting it together

The ClearML pipeline in
[`pipelines/distill/run.py`](../pipelines/distill/run.py) is exactly this flow with three
additions: configs/artifacts come from ClearML, a `DataModule` resolves datasets from
ClearML, and checkpoints are uploaded back. If you outgrow the manual setup, switch to
[Training with ClearML](./training_with_clearml.md).
