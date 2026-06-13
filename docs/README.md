# Lednik documentation

Lednik distills large transformer encoders into small, fast students — either
**Static Embeddings** (a weighted lookup table) or a tiny **Lednik Transformer**
(RoPE + RMSNorm + SwiGLU, Flash-Attention ready). These docs explain how to initialize a
student and how to distill it, with or without ClearML.

## Read in this order

1. **[Model Initialization](./model_initialization.md)**
   Create a student from a teacher: the two architectures, the factory functions, configs,
   saving/reloading, and inference. Start here regardless of how you plan to train.

2. Pick your training path:

   - **[Training without ClearML](./training_without_clearml.md)** — pure-Python distillation.
     How to build the `DistillationConfig`, instantiate the `DistillationModule`, the exact
     **data format the collator produces** (so `_base_step` can consume it), and a minimal
     `lightning.Trainer` loop.

   - **[Training with ClearML](./training_with_clearml.md)** — the production pipeline in
     `pipelines/distill/`. How models/checkpoints are loaded from the registry, how configs
     and datasets are wired, checkpoint uploading, remote queues, the online validation
     worker (Redis + Qdrant), and MTEB benchmarking.

## A note on `kostyl`

Lednik leans on **kostyl** (the author's personal ML toolkit), vendored as the
[`kostyl_toolkit/`](../kostyl_toolkit) git submodule and installed as `kostyl-toolkit[ml]`.
It provides logging, config schemas, optimizer/scheduler factories, distributed helpers, and
the PyTorch-Lightning + ClearML integrations. Every `kostyl.*` import resolves to that
submodule — initialize it with `git submodule update --init --recursive`.

## Map of the codebase

| Path | What's there |
| --- | --- |
| [`lednik/initialization/`](../lednik/initialization) | Model factory (`create_static_embeddings_model`, `create_lednik_transformer`), PCA, tokenizer customization. |
| [`lednik/models/`](../lednik/models) | `LednikModel`, `StaticEmbeddingsModel` (+ classification head), configs, outputs. |
| [`lednik/distill/`](../lednik/distill) | `DistillationModule` (Lightning), `ContrastiveCollator`, configs, losses, online validation. |
| [`lednik/emb_utils.py`](../lednik/emb_utils.py) | Teacher embedding extraction & pooling. |
| [`pipelines/distill/`](../pipelines/distill) | ClearML + Lightning distillation pipeline (`run.py`, `datamodule.py`, `configs.py`). |
| [`bench/mteb/`](../bench/mteb) | MTEB benchmark runner and model wrapper. |
| [`configs/`](../configs) | YAML configs: `distill_config.yaml`, `training_settings.yaml`, `worker_config.yaml`. |
| [`kostyl_toolkit/`](../kostyl_toolkit) | The `kostyl` submodule. |
