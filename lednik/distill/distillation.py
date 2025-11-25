from typing import cast
from typing import Literal

import lightning as L
import torch
from kostyl.utils.logging import setup_logger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import SingleDeviceStrategy
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from .dim_reduction import PCA
from .embedding_extraction import extract_embeddings
from .postprocessing import calculate_token_weights
from .postprocessing import modify_tokenizer
from lednik.distill.training.configs import TrainConfig
from lednik.distill.training.training_modules import FineTuningModule
from lednik.static_embeddings.config import StaticEmbeddingsConfig
from lednik.static_embeddings.modeling import StaticEmbeddingsModel


logger = setup_logger(fmt="only_message")


def distill_with_finetuning(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    embedding_dim: int,
    pooling: Literal["mean", "last"],
    embedding_extraction_batch_size: int,
    trainer: L.Trainer,
    train_cfg: TrainConfig,
    data: L.LightningDataModule | dict[str, torch.utils.data.DataLoader],
    dropout_p: float = 0.0,
    dtype: Literal["float32", "float16", "bfloat16"] | torch.dtype = "bfloat16",
    load_best_ckpt: bool = False,
    sif_coefficient: float | None = 1e-4,
) -> StaticEmbeddingsModel:
    """Distill static embeddings from pretrained transformer with CosineEmbeddingLoss."""
    if load_best_ckpt and trainer.checkpoint_callback is None:
        raise ValueError(
            "Checkpoint callback must be provided to load best checkpoint."
        )
    if not isinstance(trainer.strategy, SingleDeviceStrategy):
        raise ValueError("Post-training distillation only supports single device.")
    if isinstance(data, dict):
        data_container = data
    else:
        data_container = {"datamodule": data}

    model.to(dtype=dtype)  # type: ignore

    vocab_tokens_ids = [[tok_id] for tok_id in tokenizer.get_vocab().values()]
    vocab_len = len(vocab_tokens_ids)
    pad_token = int(tokenizer.pad_token_id)  # type: ignore

    embeddings_t = extract_embeddings(
        model=model,
        vocab_tokens=vocab_tokens_ids,
        pad_token=pad_token,
        pooling=pooling,
        batch_size=embedding_extraction_batch_size,
        device=trainer.strategy.root_device,
    )
    pca = PCA(n_components=embedding_dim)
    output = pca.transform(embeddings_t)
    embeddings_t = output.reduced_data.cpu()

    config_dict = {
        "vocab_size": vocab_len,
        "embedding_dim": embedding_dim,
        "dropout_p": dropout_p,
        "pad_token_id": pad_token,
        "dtype": dtype,
    }

    static_model_cfg = StaticEmbeddingsConfig.from_dict(config_dict)
    static_model = StaticEmbeddingsModel.initialize(static_model_cfg, embeddings_t)

    finetuning_module = FineTuningModule(
        train_cfg=train_cfg,
        static_model=static_model,
        teacher=model,
        tokenizer=tokenizer,
    )
    trainer.fit(model=finetuning_module, **data_container)  # type: ignore

    finetuning_module = trainer.strategy.lightning_module
    if finetuning_module is None:
        raise ValueError("Trainer strategy did not return a Lightning module.")

    finetuning_module = cast(FineTuningModule, finetuning_module)

    finetuning_module.cpu()

    if load_best_ckpt and trainer.checkpoint_callback is not None:
        checkpoint_callback = cast(ModelCheckpoint, trainer.checkpoint_callback)
        best_ckpt_path = checkpoint_callback.best_model_path
        logger.info(f"Loading best checkpoint from: {best_ckpt_path}")
        finetuning_module.load_state_dict(
            torch.load(best_ckpt_path, map_location="cpu", mmap=True)["state_dict"]
        )

    static_model = finetuning_module.static_model
    if sif_coefficient is not None:
        token_pos_weights = calculate_token_weights(
            vocab_size=static_model.config.vocab_size,
            sif_coefficient=sif_coefficient,
        )
    else:
        token_pos_weights = torch.ones(static_model.config.vocab_size)
    tokenizer = modify_tokenizer(tokenizer)

    static_model.update_pos_weights(token_pos_weights)
    static_model.add_tokenizer(tokenizer)
    return static_model


def distill(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    embedding_dim: int,
    pooling: Literal["mean", "last"],
    embedding_extraction_batch_size: int,
    device: str | torch.device,
    dtype: Literal["float32", "float16", "bfloat16"] | torch.dtype = "bfloat16",
    sif_coefficient: float | None = 1e-4,
) -> StaticEmbeddingsModel:
    """Distill static embeddings from pretrained transformer with CosineEmbeddingLoss."""
    model.to(dtype=dtype)  # type: ignore

    vocab_tokens_ids = [[tok_id] for tok_id in tokenizer.get_vocab().values()]
    vocab_len = len(vocab_tokens_ids)
    pad_token = int(tokenizer.pad_token_id)  # type: ignore

    embeddings_t = extract_embeddings(
        model=model,
        vocab_tokens=vocab_tokens_ids,
        pad_token=pad_token,
        pooling=pooling,
        batch_size=embedding_extraction_batch_size,
        device=device,
    )
    pca = PCA(n_components=embedding_dim)
    output = pca.transform(embeddings_t)
    embeddings_t = output.reduced_data.cpu()

    config_dict = {
        "vocab_size": vocab_len,
        "embedding_dim": embedding_dim,
        "dropout_p": 0.0,
        "pad_token_id": pad_token,
        "dtype": dtype,
    }

    static_model_cfg = StaticEmbeddingsConfig.from_dict(config_dict)
    static_model = StaticEmbeddingsModel.initialize(static_model_cfg, embeddings_t)

    if sif_coefficient is not None:
        token_pos_weights = calculate_token_weights(
            vocab_size=static_model.config.vocab_size,
            sif_coefficient=sif_coefficient,
        )
    else:
        token_pos_weights = torch.ones(static_model.config.vocab_size)

    tokenizer = modify_tokenizer(tokenizer)

    static_model.update_pos_weights(token_pos_weights)
    static_model.add_tokenizer(tokenizer)
    return static_model
