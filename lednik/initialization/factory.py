from typing import Any
from typing import Literal

import torch
from kostyl.utils.logging import setup_logger
from transformers import PreTrainedModel
from transformers import TokenizersBackend
from transformers.modeling_utils import local_torch_dtype

from lednik.emb_utils import extract_embeddings
from lednik.models import LednikConfig
from lednik.models import LednikModel
from lednik.models import StaticEmbeddingsConfig
from lednik.models import StaticEmbeddingsModel

from .dim_reduction import PCA
from .generic import get_torch_dtype
from .tokenizer_utils import calculate_token_weights
from .tokenizer_utils import customize_tokenizer


logger = setup_logger(fmt="only_message")


def create_static_embeddings_model(
    model: PreTrainedModel,
    tokenizer: TokenizersBackend,
    embedding_dim: int,
    pooling: Literal["mean", "last", "cls"],
    embedding_extraction_batch_size: int,
    dtype: str | torch.dtype = "float32",
    modify_tokenizer: bool = False,
    sif_coefficient: float | None = 1e-4,
    output_device: torch.device | None = None,
    **kwargs: Any,
) -> StaticEmbeddingsModel:
    """
    Create a Static Embeddings model initialized with embeddings distilled from a teacher model.

    This function extracts embeddings from a pretrained transformer model's vocabulary,
    reduces their dimensionality using PCA, and creates a static embeddings model.

    Args:
        model (`PreTrainedModel`):
            A pretrained transformer model to extract embeddings from.
        tokenizer (`TokenizersBackend`):
            The tokenizer associated with the pretrained model.
        embedding_dim (`int`):
            The target dimensionality for the output embeddings after PCA.
        pooling (`Literal["mean", "last", "cls"]`):
            The pooling strategy to use when extracting embeddings.
        embedding_extraction_batch_size (`int`):
            Batch size for extracting embeddings from the model.
        dtype (`str` | `torch.dtype`, *optional*, defaults to `"float32"`):
            The data type for model computations and output embeddings.
        modify_tokenizer (`bool`, *optional*, defaults to `False`):
            Whether to customize the tokenizer for the static model.
            > **Note:** For distillation purposes, it is recommended to keep this as `False` because modifying the
            tokenizer can affect the consistency of tokenization between the teacher and static models,
            therefore potentially impacting the quality of the distillation.
            You can modify the tokenizer after distillation if needed via `customize_tokenizer`
            from `lednik.distill.tokenizer_utils`.
        sif_coefficient (`float` | `None`, *optional*, defaults to 1e-4):
            Coefficient for Smooth Inverse Frequency (SIF) weighting. If `None`, uniform weights are used.
        output_device (`torch.device` | None, *optional*, defaults to `None`):
            The device on which to place the output model. If `None`, uses the device of the input model.
        **kwargs (`Any`):
            Additional keyword arguments for [`StaticEmbeddingsConfig`].

    Returns:
        `StaticEmbeddingsModel`:
            A static embeddings model containing the distilled embeddings with the associated tokenizer.

    Note:
        The function temporarily changes the default PyTorch dtype during execution
        and restores it before returning.

    """
    if output_device is None:
        output_device = model.device

    dtype = get_torch_dtype(dtype)
    with local_torch_dtype(dtype):
        model.to(dtype=dtype).eval()

        vocab_tokens_ids = [[tok_id] for tok_id in tokenizer.get_vocab().values()]
        vocab_len = len(vocab_tokens_ids)
        pad_token = int(tokenizer.pad_token_id)  # type: ignore

        embeddings_t = extract_embeddings(
            model=model,
            vocab_tokens=vocab_tokens_ids,
            pad_token=pad_token,
            pooling=pooling,
            batch_size=embedding_extraction_batch_size,
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
            "hidden_size": embedding_dim,
            "embedding_dropout": 0.0,
            "dtype": dtype,
            "is_tokenizer_customized": modify_tokenizer,
            **kwargs,
        }

        static_model_cfg = StaticEmbeddingsConfig.from_dict(config_dict)
        static_model = StaticEmbeddingsModel.initialize(
            static_model_cfg, embeddings_t
        ).to(device=output_device)

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
        static_model.replace_pos_weights(token_pos_weights)

        if modify_tokenizer:
            customize_tokenizer(static_model.config, tokenizer)
        static_model.add_tokenizer(tokenizer)
    return static_model


def create_lednik_transformer(
    model: PreTrainedModel,
    tokenizer: TokenizersBackend,
    model_config: LednikConfig,
    pooling: Literal["mean", "last", "cls"],
    embedding_extraction_batch_size: int,
    dtype: str | torch.dtype = "float32",
    output_device: torch.device | None = None,
) -> LednikModel:
    """
    Create a Lednik model initialized with embeddings distilled from a teacher model.

    This function extracts embeddings from a pretrained transformer model's vocabulary,
    reduces their dimensionality using PCA to match `model_config.hidden_size`,
    and initializes a new Lednik model with these embeddings.

    Args:
        model (`PreTrainedModel`):
            A pretrained transformer model to extract embeddings from.
        tokenizer (`TokenizersBackend`):
            The tokenizer associated with the pretrained model.
        model_config (`LednikConfig`):
            The configuration for the new Lednik model.
        pooling (`Literal["mean", "last", "cls"]`):
            The pooling strategy to use when extracting embeddings.
        embedding_extraction_batch_size (`int`):
            Batch size for extracting embeddings from the model.
        dtype (`str` | `torch.dtype`, *optional*, defaults to `"float32"`):
            The data type for model computations and output embeddings.
        output_device (`torch.device` | None, *optional*, defaults to `None`):
            The device on which to place the output model. If `None`, uses the device of the input model.

    Returns:
        `LednikModel`:
            A new Lednik model instance with initialized embeddings.

    Note:
        The function temporarily changes the default PyTorch dtype during execution
        and restores it before returning.

    """
    if output_device is None:
        output_device = model.device
    dtype = get_torch_dtype(dtype)

    with local_torch_dtype(dtype):
        model.to(dtype=dtype).eval()

        vocab_tokens_ids = [[tok_id] for tok_id in tokenizer.get_vocab().values()]
        vocab_size = len(vocab_tokens_ids)
        if model_config.vocab_size != vocab_size:
            logger.warning(
                f"Passed config's vocab size {model_config.vocab_size} does not match tokenizer vocab size {vocab_size}. "
                "Updating config."
            )
            model_config.vocab_size = vocab_size
        pad_token = int(tokenizer.pad_token_id)  # type: ignore
        if pad_token != model_config.pad_token_id:
            logger.warning(
                f"Passed config's pad token id {model_config.pad_token_id} does not match tokenizer pad token id {pad_token}. "
                "Updating config."
            )
            model_config.pad_token_id = pad_token

        embeddings_t = extract_embeddings(
            model=model,
            vocab_tokens=vocab_tokens_ids,
            pad_token=pad_token,
            pooling=pooling,
            batch_size=embedding_extraction_batch_size,
        )
        pca = PCA(n_components=model_config.hidden_size)
        output = pca.transform(embeddings_t)
        embeddings_t = output.reduced_data
        embeddings_t = embeddings_t.to(output_device)
        with torch.device("meta"):
            model = LednikModel(config=model_config)
        model.to_empty(device=output_device)
        model.init_weights()
        model.replace_embeddings(embeddings_t)
    return model
