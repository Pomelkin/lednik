# Lednik and the JMLC criteria

This directory maps the repository onto the evaluation criteria of the
[Junior ML Contest](https://ai.itmo.ru/junior_ml_contest). Each page covers one
criterion and links to the code, configs and committed benchmark records behind it.

Project summary: Lednik distills large Russian embedding encoders into small, fast
students. The teacher, [`deepvk/USER-bge-m3`](https://huggingface.co/deepvk/USER-bge-m3)
(359M parameters), sustains ~5 requests/s on an RTX 3080 at ~2.3k-token payloads. The
distilled 56M student keeps ~82% of the teacher's RuMTEB score, holds 100 requests/s on
the same GPU with p99 = 216 ms, and vectorizes a billion tokens for 12.7 ₽ instead of
233 ₽. Result tables: [data_science.md](./data_science.md) and the root
[README](../../README.md#results).

| Criterion                 | Page                                | Contents                                                                                                                                       |
| ------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Development & engineering | [engineering.md](./engineering.md)   | Git workflow, CI, tests, quality gates, Docker, and the ClearML loop: model registry, dataset versioning, remote execution, online validation. |
| Data Science              | [data_science.md](./data_science.md) | Distillation method, training data, validation protocol, measured results, known limitations.                                                  |
| AI tools in development   | [ai_tooling.md](./ai_tooling.md)     | Agentic coding in the workflow; LLM-based synthetic data generation inside the pipeline.                                                       |
| Product thinking          | [product.md](./product.md)           | The serving-cost problem, target users, alternatives, measured impact in GPUs and rubles.                                                      |

The fifth criterion, motivation, belongs to the personal submission, not the repository.

Usage guides for the framework itself:
- [model initialization](../model_initialization.md)
- [training without ClearML](../training_without_clearml.md)
- [training with ClearML](../training_with_clearml.md)
- [serving](../usage.md).
