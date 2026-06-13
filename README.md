# Lednik

Lednik distills large transformer encoders into small, fast students — either
**Static Embeddings** (a weighted token lookup table) or a tiny **Lednik Transformer**
(RoPE + RMSNorm + SwiGLU/GeGLU, Flash-Attention ready). It provides the model definitions,
an initialization factory that seeds students from a teacher, a PyTorch-Lightning
distillation trainer, and a ready-made [ClearML](https://clear.ml/) pipeline.

## Features

- **Static Embeddings distillation** — compress a teacher's vocabulary into a static
  embedding matrix via PCA and Smooth-Inverse-Frequency (SIF) token weighting.
- **Lednik Transformer distillation** — initialize a small encoder from a teacher and
  distill knowledge into it (contrastive + regression objective).
- **Two training paths** — a pure-Python loop, or a full ClearML + Lightning pipeline with
  artifact loading, config syncing, remote queues and online validation.
- **Efficient inference** — Flash-Attention / varlen SDPA, Liger kernels, low-precision
  optimizers.

## Documentation

Full guides live in [`docs/`](./docs):

- **[Model Initialization](./docs/model_initialization.md)** — create a student from a
  teacher (the two architectures, factory functions, configs, save/reload, inference).
- **[Training without ClearML](./docs/training_without_clearml.md)** — build the config,
  instantiate the training module, the collator data format, and a minimal `Trainer` loop.
- **[Training with ClearML](./docs/training_with_clearml.md)** — the `pipelines/distill/`
  pipeline: checkpoint loading, config/dataset wiring, checkpoint uploads, remote queues,
  the online validation worker, and MTEB benchmarking.

## Project structure

```
lednik/
├── lednik/                 # core library
│   ├── initialization/     # model factory (create_* fns), PCA, tokenizer utils
│   ├── models/             # LednikModel, StaticEmbeddingsModel (+ configs, outputs)
│   ├── distill/            # DistillationModule, collator, configs, losses, validation
│   ├── emb_utils.py        # teacher embedding extraction & pooling
│   └── dist_utils.py       # FSDP/DDP helpers, distributed embedding gather
├── pipelines/distill/      # ClearML + Lightning distillation pipeline
├── bench/mteb/             # MTEB benchmark runner + model wrapper
├── eda_utils/              # synthetic data generation utilities
├── configs/                # YAML configs (distill / training_settings / worker)
├── kostyl_toolkit/         # git submodule: the `kostyl` ML toolkit (used throughout)
└── docs/                   # documentation
```

> **`kostyl`.** Lednik depends on **kostyl-toolkit**, the author's personal ML toolkit,
> vendored as the [`kostyl_toolkit/`](./kostyl_toolkit) git submodule and installed as
> `kostyl-toolkit[ml]`. Every `kostyl.*` import resolves to it.

## Installation

```bash
# Clone with the kostyl submodule
git clone --recurse-submodules <repo-url>
# or, if already cloned:
git submodule update --init --recursive

# Install with uv (the project's package manager)
uv sync                       # core + default groups (dev, distill)
uv sync --group flash-attn    # optional: Flash-Attention for LednikModel on GPU
uv sync --group bench         # optional: MTEB benchmarking
```

Python ≥ 3.13 is required (see [`pyproject.toml`](./pyproject.toml)).

## Quickstart

Initialize a Lednik Transformer student from a teacher:

```python
from transformers import AutoModel, AutoTokenizer
from lednik.models import LednikConfig
from lednik.initialization.factory import create_lednik_transformer

teacher = AutoModel.from_pretrained("deepvk/USER-bge-m3").to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-bge-m3")

config = LednikConfig(
    hidden_size=384,
    num_hidden_layers=2,
    num_attention_heads=6,
    intermediate_size=1152,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
)

student = create_lednik_transformer(
    model=teacher,
    tokenizer=tokenizer,
    model_config=config,
    pooling="mean",
    embedding_extraction_batch_size=256,
)
student.save_pretrained("weights/lednik_base")
tokenizer.save_pretrained("weights/lednik_base")
```

Then distill it — see [Training without ClearML](./docs/training_without_clearml.md) for the
pure-Python loop, or run the ClearML pipeline:

```bash
python -m pipelines.distill.run                              # local
python -m pipelines.distill.run --remote-execution-queue q   # remote ClearML agent
```

## Models

- **`LednikModel`** — a small encoder transformer with Rotary Positional Embeddings,
  RMSNorm, SwiGLU/GeGLU MLPs and Flash-Attention / torch-varlen backends. Returns
  `last_hidden_state` and mean-pooled `sentence_embeddings`.
- **`StaticEmbeddingsModel`** — maps tokens directly to static vectors with per-token SIF
  weights and mean pooling; no attention. Ideal for ultra-low-latency / CPU inference. A
  `StaticEmbeddingsForSequenceClassification` head is also available.
