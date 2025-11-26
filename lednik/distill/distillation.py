from typing import Literal

import lightning as L
import torch
from kostyl.utils.logging import setup_logger
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from .dim_reduction import PCA
from .embedding_extraction import extract_embeddings
from .postprocessing import calculate_token_weights
from .postprocessing import customize_tokenizer
from lednik.distill.training.configs import TrainConfig
from lednik.distill.training.training_modules import FineTuningModule
from lednik.static_embeddings.config import StaticEmbeddingsConfig
from lednik.static_embeddings.modeling import StaticEmbeddingsModel


logger = setup_logger(fmt="only_message")


def finetune(
    teacher: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    static_model: StaticEmbeddingsModel,
    trainer: L.Trainer,
    train_cfg: TrainConfig,
    data: L.LightningDataModule | dict[str, torch.utils.data.DataLoader],
) -> StaticEmbeddingsModel:
    """Distill static embeddings from pretrained transformer with CosineEmbeddingLoss."""
    if isinstance(data, dict):
        data_container = data
    else:
        data_container = {"datamodule": data}

    finetuning_module = FineTuningModule(
        train_cfg=train_cfg,
        static_model=static_model,
        teacher=teacher,
        tokenizer=tokenizer,
    )
    trainer.fit(model=finetuning_module, **data_container)  # type: ignore

    return static_model


def distill(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    embedding_dim: int,
    pooling: Literal["mean", "last", "cls"],
    embedding_extraction_batch_size: int,
    device: str | torch.device,
    dtype: str | torch.dtype = "bfloat16",
    modify_tokenizer: bool = True,
    sif_coefficient: float | None = 1e-4,
) -> StaticEmbeddingsModel:
    """Distill static embeddings from pretrained transformer with CosineEmbeddingLoss."""
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(dtype, str):
        dtype_ = getattr(torch, dtype)
    else:
        dtype_ = dtype

    original_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype_)
    model.to(dtype=dtype_, device=device)  # type: ignore

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
    embeddings_t = output.reduced_data

    logger.info(
        f"Explained variance ratio by PCA: {pca.explained_variance_ratio.sum():.4f}"
    )

    config_dict = {
        "vocab_size": vocab_len,
        "pad_token_id": pad_token,
        "embedding_dim": embedding_dim,
        "dropout_p": 0.0,
        "dtype": dtype_,
    }

    static_model_cfg = StaticEmbeddingsConfig.from_dict(config_dict)
    static_model = (
        StaticEmbeddingsModel.initialize(static_model_cfg, embeddings_t)
        .to(device)  # pyright: ignore[reportArgumentType]
        .eval()
    )

    if sif_coefficient is not None:
        token_pos_weights = calculate_token_weights(
            vocab_size=static_model.config.vocab_size,
            sif_coefficient=sif_coefficient,
        )
    else:
        token_pos_weights = torch.ones(static_model.config.vocab_size)
    token_pos_weights = token_pos_weights.unsqueeze(-1).to(
        device=static_model.device, dtype=static_model.dtype
    )
    static_model.update_pos_weights(token_pos_weights)

    if modify_tokenizer:
        tokenizer = customize_tokenizer(tokenizer)
    static_model.add_tokenizer(tokenizer)

    torch.set_default_dtype(original_default_dtype)
    return static_model
