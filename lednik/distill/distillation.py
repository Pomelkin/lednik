from typing import cast
from typing import Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies import ModelParallelStrategy
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from .dim_reduction import PCA
from .embedding_extraction import extract_embeddings
from .postprocessing import calculate_token_weights
from .postprocessing import modify_tokenizer
from lednik.distill.train.configs import TrainConfig
from lednik.distill.train.training_modules import PostTrainModule
from lednik.static_embeddings.config import StaticEmbeddingsConfig
from lednik.static_embeddings.modeling import StaticEmbeddingsModel
from lednik.static_embeddings.modeling import StaticEmbeddingsModelForPostTraining
from lednik.utils.logging import setup_logger

logger = setup_logger(fmt="only_message")


def distill_with_post_training(
    trainer: L.Trainer,
    train_cfg: TrainConfig,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    embedding_dim: int,
    pooling: Literal["mean", "last"],
    embedding_extraction_batch_size: int,
    embedding_extraction_device: str | torch.device,
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
    if trainer.strategy in [FSDPStrategy, DeepSpeedStrategy, ModelParallelStrategy]:
        raise ValueError(
            "Distillation with post-training is not supported with FSDP, DeepSpeed, or ModelParallel strategies."
        )

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
        device=embedding_extraction_device,
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
    static_model_for_post_train = StaticEmbeddingsModelForPostTraining.initialize(
        static_model_cfg, embeddings_t
    )

    training_module = PostTrainModule(
        train_cfg=train_cfg,
        static_model=static_model_for_post_train,
        teacher=model,
        tokenizer=tokenizer,
    )
    trainer.fit(
        model=training_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    training_module = trainer.strategy.lightning_module
    if training_module is None:
        raise ValueError("Trainer strategy did not return a Lightning module.")

    training_module = cast(PostTrainModule, training_module)

    training_module.cpu()

    if load_best_ckpt and trainer.checkpoint_callback is not None:
        checkpoint_callback = cast(ModelCheckpoint, trainer.checkpoint_callback)
        best_ckpt_path = checkpoint_callback.best_model_path
        logger.info(f"Loading best checkpoint from: {best_ckpt_path}")
        training_module.load_state_dict(
            torch.load(best_ckpt_path, map_location="cpu", mmap=True)["state_dict"]
        )

    static_model_for_post_train = training_module.static_model
    if sif_coefficient is not None:
        token_pos_weights = calculate_token_weights(
            vocab_size=static_model_for_post_train.config.vocab_size,
            sif_coefficient=sif_coefficient,
        )
    else:
        token_pos_weights = torch.ones(static_model_for_post_train.config.vocab_size)
    tokenizer = modify_tokenizer(tokenizer)
    static_model = static_model_for_post_train.to_static_model(
        token_pos_weights, tokenizer=tokenizer
    )
    return static_model
