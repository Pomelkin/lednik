from typing import Literal

import lightning as L
import torch
from clearml import Task
from kostyl.utils.logging import setup_logger
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from lednik.distill.training.configs import TrainConfig
from lednik.distill.training.training_modules import FineTuningModule
from lednik.static_embeddings.config import StaticEmbeddingsConfig
from lednik.static_embeddings.modeling import StaticEmbeddingsModel

from .dim_reduction import PCA
from .extraction_utils import extract_embeddings
from .postprocessing import calculate_token_weights
from .postprocessing import customize_tokenizer


logger = setup_logger(fmt="only_message")


def finetune(
    teacher: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    static_model: StaticEmbeddingsModel,
    trainer: L.Trainer,
    train_cfg: TrainConfig,
    data: L.LightningDataModule | dict[str, torch.utils.data.DataLoader],
    task: Task | None = None,
) -> StaticEmbeddingsModel:
    """
    Distill static embeddings from a pretrained transformer using CosineEmbeddingLoss.

    This function orchestrates the distillation process by wrapping the teacher model,
    static model, and training configuration into a `FineTuningModule` (a LightningModule)
    and executing the training loop via the provided PyTorch Lightning Trainer.

    Args:
        teacher (PreTrainedModel): The source transformer model to distill knowledge from.
        tokenizer (PreTrainedTokenizerBase): The tokenizer corresponding to the teacher model.
        static_model (StaticEmbeddingsModel): The target model that will learn static embeddings.
        trainer (L.Trainer): The PyTorch Lightning Trainer instance to manage the training process.
        train_cfg (TrainConfig): Configuration object containing hyperparameters for training.
        data (L.LightningDataModule | dict[str, torch.utils.data.DataLoader]): The training data,
            provided either as a LightningDataModule or a dictionary of DataLoaders.
        task (Task | None, optional): An optional specific task configuration defining how
            embeddings are processed or evaluated. Defaults to None.

    Returns:
        StaticEmbeddingsModel: The trained static embeddings model after the distillation process.

    """
    if isinstance(data, dict):
        data_container = data
    else:
        data_container = {"datamodule": data}

    finetuning_module = FineTuningModule(
        train_cfg=train_cfg,
        static_model=static_model,
        teacher=teacher,
        tokenizer=tokenizer,
        task=task,
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
    """
    Distill static embeddings from pretrained transformer with CosineEmbeddingLoss.

    This function extracts embeddings from a pretrained transformer model's vocabulary,
    reduces their dimensionality using PCA, and creates a static embeddings model.

    Args:
        model: A pretrained transformer model to extract embeddings from.
        tokenizer: The tokenizer associated with the pretrained model.
        embedding_dim: The target dimensionality for the output embeddings after PCA.
        pooling: The pooling strategy to use when extracting embeddings.
            Must be one of "mean", "last", or "cls".
        embedding_extraction_batch_size: Batch size for extracting embeddings from the model.
        device: The device to run computations on (e.g., "cuda", "cpu").
        dtype: The data type for model computations. Defaults to "bfloat16".
        modify_tokenizer: Whether to customize the tokenizer for the static model.
            Defaults to True.
        sif_coefficient: Coefficient for Smooth Inverse Frequency (SIF) weighting.
            If None, uniform weights are used. Defaults to 1e-4.

    Returns:
        StaticEmbeddingsModel: A static embeddings model containing the distilled
            embeddings with the associated tokenizer.

    Note:
        The function temporarily changes the default PyTorch dtype during execution
        and restores it before returning.

    """
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(dtype, str):
        dtype_ = getattr(torch, dtype)
    else:
        dtype_ = dtype

    original_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype_)
    model.to(dtype=dtype_, device=device).eval()  # type: ignore

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
        "embedding_dropout": 0.0,
        "dtype": dtype_,
        "tokenizer_modified": modify_tokenizer,
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
