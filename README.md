# Static Embeddings Distillation

This repository distills transformer knowledge into a lightweight static embedding model and optionally fine-tunes it with task-specific supervision. The core logic lives in `lednik/distill/distillation.py`, which exposes two entry points: `distill` for building a frozen embedding space and `finetune` for aligning it with downstream labels.

## Static Model (`lednik/static_embeddings`)

- `StaticEmbeddingsConfig` defines tensor shapes, vocabulary metadata, and optional tokenizer customization flags.
- `StaticEmbeddingsModel` wraps a learnable embedding table with dropout, positional weighting, and helper methods for attaching a tokenizer or updating token weights.
- By default the model stores embeddings reduced via PCA and keeps PyTorch on the device/dtype passed to `distill` and `finetune`.

## `distill` function

````python
from lednik.distill.distillation import distill

static_model = distill(
	model=teacher,
	tokenizer=teacher_tokenizer,
	embedding_dim=300,
	pooling="mean",  # "mean", "last", or "cls"
	embedding_extraction_batch_size=256,
	device="cuda",
	dtype="bfloat16",
	modify_tokenizer=True,
	sif_coefficient=1e-4,
)
````

`distill` workflow:

1. Extract transformer hidden states for every vocabulary token with the requested pooling strategy (`extract_embeddings`).
2. Run PCA (`lednik/distill/dim_reduction.py::PCA`) to shrink representations to `embedding_dim` while logging explained variance.
3. Build a `StaticEmbeddingsModel` with the reduced vectors and optional Smooth Inverse Frequency weights (`calculate_token_weights`).
4. Optionally customize the tokenizer to align with the static model (`customize_tokenizer`).

Use this when you only need fast lookups (e.g., retrieval, clustering) or as an initialization checkpoint for later supervised training.

## `finetune` function

````python
from lednik.distill.distillation import finetune
from lednik.distill.training.configs import FinetuningConfig

finetuned_model = finetune(
	teacher=teacher,
	tokenizer=teacher_tokenizer,
	static_model=static_model,
	trainer=lightning_trainer,
	train_cfg=FinetuningConfig(...),
	data=lightning_datamodule_or_dataloaders,
	task=optional_clearml_task,
)
````

- Wraps everything into `FineTuningModule` (`lednik/distill/training/training_modules/finetuning.py`).
- Leverages PyTorch Lightning’s `Trainer.fit` with either a `LightningDataModule` or explicit loader dict.
- Maintains cosine embedding loss training by default; customize via `FinetuningConfig`.

Use `finetune` after `distill` when a downstream dataset exists (classification, similarity, etc.). The function returns the updated `StaticEmbeddingsModel` so its tokenizer, PCA space, and positional weights stay in sync.

## Typical pipeline

````python
static_model = distill(...)
finetuned_static_model = finetune(..., static_model=static_model, ...)

torch.save(finetuned_static_model.state_dict(), "static_embeddings.pt")
````

## Tips

- Make sure the tokenizer has a valid `pad_token_id`; `distill` relies on it during batch inference.
- When `sif_coefficient` is `None`, token weights default to 1.0 (uniform frequency assumption).
- `distill` temporarily changes the default PyTorch dtype—avoid running other model code in parallel threads that depends on `torch.get_default_dtype()`.
- Use Lightning checkpoints to resume fine-tuning; `FineTuningModule` is Lightning-native, so `Trainer.fit` handles logging and callbacks out of the box.
