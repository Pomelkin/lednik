from typing import Literal

import lightning as L
import torch
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from .dim_reduction import PCA
from .embedding_extraction import extract_embeddings
from lednik.static_embeddings.config import StaticEmbeddingsConfig
from lednik.static_embeddings.modeling import StaticEmbeddingsModelForPostTraining
from lednik.utils.logging import setup_logger

logger = setup_logger(fmt="only_message")


def distill_with_postraining(
    trainer: L.Trainer,
    lr: float,
    warmup_lr: float,
    warmup_steps: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    embedding_dim: int,
    pooling: Literal["mean", "last"],
    embedding_extraction_batch_size: int,
    embedding_extraction_device: str | torch.device,
    dropout_p: float = 0.0,
    dtype: Literal["float32", "float16", "bfloat16"] | torch.dtype = "bfloat16",
) -> None:
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
        device=embedding_extraction_device,
    )
    pca = PCA(n_components=embedding_dim)
    output = pca.transform(embeddings_t)
    embeddings_t = output.reduced_data.cpu()

    static_embeddings = torch.nn.Embedding(
        num_embeddings=vocab_len,
        embedding_dim=embedding_dim,
        padding_idx=pad_token,
        device="cpu",
        dtype=dtype,
    )
    static_embeddings.weight.data = embeddings_t

    config_dict = {
        "vocab_size": vocab_len,
        "embedding_dim": embedding_dim,
        "dropout_p": dropout_p,
        "pad_token_id": pad_token,
        "dtype": dtype,
    }

    static_model_cfg = StaticEmbeddingsConfig.from_dict(config_dict)
    static_model = StaticEmbeddingsModelForPostTraining(config=static_model_cfg)
    static_model.update_embeddings(static_embeddings)
    return
