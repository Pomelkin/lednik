# Lednik documentation

Lednik distills large transformer encoders into small, fast students — a **Static
Embeddings** lookup table, a tiny **Lednik Transformer** (RoPE + RMSNorm + SwiGLU, gated
attention, Flash-Attention/varlen ready), or a **hybrid** variant that mixes in
bidirectional Gated DeltaNet (linear attention) blocks. These docs explain how to
initialize a student, how to distill it (with or without ClearML), and how to serve and
benchmark the result.

## Read in this order

1. **[Model Initialization](./model_initialization.md)**
   Create a student from a teacher: the architectures, the factory functions, configs,
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

3. **[Usage](./usage.md)**
   Use the trained student: `AutoLednikModel` checkpoint loading, the LitServe embedding
   server (request protocol, scaling knobs), and Docker/MPS deployment.

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
| [`lednik/models/`](../lednik/models) | `LednikModel`, `StaticEmbeddingsModel` (+ classification head), configs, outputs, `AutoLednikModel` and the model/config registries. |
| [`lednik/distill/`](../lednik/distill) | `DistillationModule` (Lightning), `ContrastiveCollator`, configs, losses, online validation. |
| [`lednik/serving/`](../lednik/serving) | The LitServe embedding server (`LednikServer`, `EmbedRequest`). |
| [`lednik/emb_utils.py`](../lednik/emb_utils.py) | Teacher embedding extraction & pooling. |
| [`lednik/path_utils.py`](../lednik/path_utils.py) | `determine_path`: ClearML ID / HF repo / local path resolution. |
| [`pipelines/distill/`](../pipelines/distill) | ClearML + Lightning distillation pipeline (`run.py`, `datamodule.py`, `configs.py`). |
| [`bench/mteb_testing/`](../bench/mteb_testing) | RuMTEB benchmark runner and model wrapper. |
| [`bench/forward_testing/`](../bench/forward_testing) | Pure forward-pass bench: `do_bench` latency, tokens/sec, VRAM. |
| [`bench/load_testing/`](../bench/load_testing) | Open-/closed-loop HTTP load generator for the serving stack. |
| [`docker/`](../docker) | Serving / training / flash-attention builder images (+ [`docker-compose.yaml`](../docker-compose.yaml)). |
| [`configs/`](../configs) | YAML configs: `training_settings.yaml`, `worker_config.yaml`. |
| [`kostyl_toolkit/`](../kostyl_toolkit) | The `kostyl` submodule. |
