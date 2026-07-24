# AI tools in development

Two distinct uses: AI assistants in the development workflow, and LLMs working inside
the data pipeline.

## Assistants in the workflow

The code is written by the author; assistants have defined roles around it.

- **Claude Code as an advisor.** Not a code generator — a consultant over the real
  repo: design discussions before building a component, targeted optimization questions
  about specific pieces of code (fusing the forward and backward GDN passes into one
  kernel call, serving bottlenecks, etc.), and debugging help. Experiments and
  hyperparameter choices are the author's own. Project context (goals, measurement
  protocols, past decisions) persists in the agent's memory between sessions.
- **Documentation.** The guides in [`docs/`](../) are written by Claude Code from the
  code and the committed benchmark records, then reviewed by the author.
- **GitHub Copilot** in next-edit-suggestion mode, as smart autocomplete for mechanical
  edits.

Assistant output has no separate lane into the codebase: the same pre-commit hooks
(ruff, `ty`), the same 31 tests in CI, and the same committed benchmark records apply to
every change regardless of where it originated.

## LLMs inside the pipeline

Part of the distillation pairs is synthetic.
[`eda_utils/synt_generation`](../../eda_utils/synt_generation) generates texts through
any OpenAI-compatible endpoint: concurrent requests via a thread pool, retries with
backoff (tenacity), checkpoint callbacks so long generation runs survive interruption,
polars DataFrames in and out. Generated pairs then go through the same offline step as
real data — teacher embeddings are precomputed and the rows enter the versioned ClearML
datasets.

The distillation setup itself keeps the loop cheap: the LLM/teacher cost is paid once,
offline (synthetic generation, teacher-embedding precompute), and training consumes only
stored vectors — see [data_science.md](./data_science.md#distillation-objective).
