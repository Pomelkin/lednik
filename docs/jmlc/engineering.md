# Engineering

How the project is built, tested, shipped and tracked. Everything below is in the
repository; commands are copy-pasteable.

## Repository and workflow

- One repo, clear boundaries: [`lednik/`](../../lednik) (library),
  [`pipelines/distill/`](../../pipelines/distill) (ClearML training pipeline),
  [`bench/`](../../bench) (three benchmark suites), [`lednik/serving/`](../../lednik/serving)
  (embedding server), [`docker/`](../../docker), [`configs/`](../../configs).
- Reusable ClearML/Lightning glue lives in a separate library,
  [`kostyl-toolkit`](https://github.com/Pomelkin/kostyl-toolkit), vendored as a git
  submodule and installed as a uv workspace member. Lednik keeps only project-specific
  code.
- 316 commits since October 2025, conventional-commit messages with scopes
  (`fix(serving)`, `perf(serving)`, `ci(uv)`, `build(docker)`). Work goes through
  `develop`; pull requests into `main` are gated by CI.

## Toolchain

- Python 3.13, [uv](https://docs.astral.sh/uv/) as the package manager, committed
  `uv.lock`, dependency groups per concern: `distill`, `serving`, `bench`, `fast-attn`,
  `test`, `dev`. CI and Docker install only the groups they need.
- `flash-attn` and `causal-conv1d` are installed from local wheels (`wheels/`), built once
  in a dedicated image ([`docker/buildFlashAttention.Dockerfile`](../../docker/buildFlashAttention.Dockerfile))
  instead of being compiled on every environment setup.

## Quality gates

- **Lint/format**: ruff with ~40 rule families enabled — bandit (`S`), bugbear (`B`),
  pyupgrade (`UP`), performance (`PERF`), mccabe complexity ≤ 10, Google-style docstring
  checks. Config in [`pyproject.toml`](../../pyproject.toml).
- **Type checking**: [`ty`](https://github.com/astral-sh/ty) runs as a pre-commit hook.
- **Pre-commit** ([config](../../.pre-commit-config.yaml)): ruff lint+format, pyupgrade,
  codespell, `uv-lock` sync, hadolint for Dockerfiles, dclint for the compose file,
  private-key detection.
- **Tests**: 31 tests in [`tests/`](../../tests) covering the load-bearing seams —
  checkpoint class resolution (`AutoLednikModel`), the distillation collator, config
  schemas, `determine_path` resolution, the static model, and varlen unpadding.
- **CI** ([`.github/workflows/tests.yml`](../../.github/workflows/tests.yml)): on every PR
  to `main` — recursive submodule checkout, cached uv, `uv sync --frozen`, pytest. The
  frozen flag makes CI fail if the lockfile drifts from `pyproject.toml`.

## MLOps: ClearML end to end

The training pipeline ([`pipelines/distill/`](../../pipelines/distill), guide:
[training with ClearML](../training_with_clearml.md)) resolves every artifact by ID and
writes every result back:

- **Model registry.** Teacher, student and tokenizer are ClearML model IDs in
  [`configs/training_settings.yaml`](../../configs/training_settings.yaml). The pipeline
  downloads them at start; nothing is passed around by hand.
- **Dataset versioning.** Training data are ClearML Datasets (name → ID mapping in the
  same config). The `DataModule` downloads them in parallel, globs `train/`/`val/`
  splits, drops columns the collator does not use, and concatenates the rest.
- **Config syncing.** The YAML config is attached to the task via `connect_as_file`; for
  remote runs, edits made in the ClearML UI override the local file.
- **Remote execution.** `--remote-execution-queue <queue>` re-enqueues the same task on a
  ClearML agent; the local process exits.
- **Checkpoint upload.** Improved checkpoints (`upload_strategy="best"`) are pushed back
  to the registry as new model versions with config attached, so any checkpoint is
  loadable by ID immediately after the epoch that produced it.
- **One resolver everywhere.** [`determine_path`](../../lednik/path_utils.py) accepts a
  ClearML model ID, an HF Hub repo id, or a local path, and returns a validated local
  path. The serving CLI and all three benchmark CLIs pass `--model`/`--tokenizer` through
  it, so a model trained in ClearML is served and benchmarked by its ID with no manual
  download step.

### Online validation with a detachable worker

Embedding-quality metrics (KNN accuracy, logistic-regression probe, MRR over Qdrant) are
too slow for the training step, so they run at the end of each validation epoch (as
frequent as `trainer.val_check_interval` makes them — up to several per training epoch):
the training process dispatches embeddings as a `ValidationContract`
([`lednik/distill/validation/`](../../lednik/distill/validation)). The consumer is
deployable two ways:

- **Separate worker.** With `redis` configured, contracts go to a Redis stream; a
  standalone worker (`uv run python -m lednik.distill.validation.worker`) — on the same host or
  a different one — consumes the stream, computes metrics, and logs scalars back to the
  originating ClearML task.
- **Local fallback.** Without Redis (or if dispatch fails mid-run), the same
  `EvaluationRunner` executes in-process, no infrastructure required. The repo default
  config runs in this mode.

## Serving

[`lednik/serving/server.py`](../../lednik/serving/server.py), built on LitServe; details
in the [usage guide](../usage.md):

- Dynamic batching that pads to the batch maximum rounded to a multiple of 8, not to the
  sequence-length limit, preserving the model's varlen fast path.
- Request validation via pydantic in the API processes: malformed requests return 422
  without touching GPU workers.
- Scaling knobs: inference workers per GPU (with CUDA MPS support), parallel HTTP
  processes, ZMQ transport instead of manager queues.
- A pre-tokenized `token_ids` request path, so load tests measure the model rather than
  the tokenizer.

Measured behavior under load is in
[data_science.md](./data_science.md#load-testing-the-served-model).

## Docker

Three images ([`docker/`](../../docker)) and a compose file with profiles
([`docker-compose.yaml`](../../docker-compose.yaml)): `serving` (CUDA 12.8, installs
`serving`+`fast-attn` groups, entrypoint is the server), `training` (interactive,
host-network, pinned GPUs), and the flash-attention wheel builder. Infrastructure
services — Redis for validation dispatch, Qdrant for MRR — are compose services under the
`training` profile. Containers build with `HOST_UID`/`HOST_GID`, so files created in
mounted volumes belong to the host user; `/tmp/nvidia-mps` is mounted into serving so
workers attach to a host MPS daemon when one is running.

## Reproducibility

- Every benchmark writes one JSON line per run with the full setup (GPU, batch size,
  sequence-length range, dtype, seed) next to the numbers; records are committed under
  `bench/*/results/metrics.jsonl`.
- Training runs are seeded (`seed_everything(42)`) and fully described by one YAML file
  that ClearML versions per task.
- The tables in the root README and in [data_science.md](./data_science.md) are taken
  from those committed records.
