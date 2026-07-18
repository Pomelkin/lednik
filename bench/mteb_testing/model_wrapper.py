from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from kostyl.utils import setup_logger
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.types import BatchedInput
from mteb.types import PromptType
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import SentencePieceBackend
from transformers import TokenizersBackend

from lednik.models import LednikModelOutput
from lednik.models import LednikPreTrainedModel
from lednik.models import StaticEmbeddingsModel
from lednik.models import StaticEmbeddingsOutput


logger = setup_logger(name="mteb/model_wrapper.py", fmt="default")


def _get_tensor_with_spec_tok_ids(
    tokenizer: SentencePieceBackend | TokenizersBackend,
) -> Tensor:
    special_tokens = tokenizer.special_tokens_map.values()
    return torch.tensor(
        [tokenizer.convert_tokens_to_ids(token) for token in special_tokens],
        dtype=torch.long,
    )


def _select_optimal_device() -> torch.device:
    """Moves the model to the optimal device and data type for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def _select_dtype(device: torch.device) -> torch.dtype:
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    return dtype


class MTEBModelWrapper:
    """A thin wrapper that adapts the model to the MTEB `encode` API.

    This wrapper unifies a few model types under the same interface:
    - `StaticEmbeddingsModel`: the model performs its own pooling and returns
        `sentence_embeddings`.
    - `LednikModel` / other `PreTrainedModel`: the wrapper applies a pooling strategy
        via `get_sentence_embedding`.

    Notes:
    - For `StaticEmbeddingsModel`, the attention mask is additionally zeroed out for
        special tokens so they do not affect pooling.
    - MTEB may call `encode` on an empty split; in that case the wrapper returns an
        empty `(0, dim)` array/tensor.
    """

    def __init__(
        self,
        model: LednikPreTrainedModel,
        tokenizer: SentencePieceBackend | TokenizersBackend,
        model_id: str | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_length: int | None = 512,
        normalize_default: bool = True,
        model_name_for_meta: str | None = None,  # for ModelMeta.name
    ) -> None:
        """Create the wrapper.

        Args:
            model: A model that produces token embeddings (or sentence embeddings in
                case of `StaticEmbeddingsModel`).
            tokenizer: HuggingFace tokenizer (`SentencePieceBackend` or
                `TokenizersBackend`).
            model_id: Optional model identifier for metadata (e.g. ClearML ID or HF repo name).
            device: Inference device. If not provided, selected automatically.
            dtype: Model dtype. If not provided, selected automatically (prefers
                `bfloat16` on CUDA when supported).
            max_length: Max sequence length for tokenization.
            normalize_default: Default L2-normalization behavior for `encode`.
            model_name_for_meta: Optional custom name for `ModelMeta.name`. If not provided, defaults to "Lednik/" + model class name.
        """
        self.model = model
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.normalize_default = normalize_default
        self.special_tokens = _get_tensor_with_spec_tok_ids(tokenizer)
        self.model_id = model_id
        self.model_name_for_meta = model_name_for_meta
        self._setup(device=device, dtype=dtype)
        return

    def _setup(
        self, device: str | torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        if device is None:
            device = _select_optimal_device()
        else:
            device = (
                device if isinstance(device, torch.device) else torch.device(device)
            )

        if dtype is None:
            dtype = _select_dtype(device)

        self.model.to(device=device, dtype=dtype)  # ty:ignore[missing-argument]
        self.model.eval()

        max_model_length = getattr(
            self.model.config,
            "max_position_embeddings",
            int(self.tokenizer.model_max_length),
        )

        self.max_length = (
            min(self.max_length, max_model_length)
            if self.max_length is not None
            else max_model_length
        )

        logger.info(
            "MTEBModelWrapper setup complete, current configuration:\n"
            f" - Model {type(self.model).__name__} is on device {device} with dtype {dtype}. "
            f" - Tokenizer max_length={self.tokenizer.model_max_length}, wrapper max_length={self.max_length}."
        )
        return

    @property
    def device(self) -> torch.device:
        """The current device of the underlying model."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """The current dtype of the underlying model parameters."""
        return next(self.model.parameters()).dtype

    @torch.inference_mode()
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool | None = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        truncate_dim: int | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray | Tensor:
        """Encode texts into sentence embeddings.

        MTEB calls this method with a `DataLoader` yielding batches. This wrapper also
        supports legacy calls where `inputs` is a single string or a list of strings.

        Args:
            inputs: Text source. Typically a `DataLoader` yielding a dict with a `"text"`
                field (string or list/tuple of strings), but simpler formats are supported.
            task_metadata: MTEB task metadata (part of the API; not used directly).
            hf_split: HF/MTEB parameter (not used directly).
            hf_subset: HF/MTEB parameter (not used directly).
            prompt_type: MTEB parameter (not used directly).
            batch_size: Batch size for tokenization and inference.
            normalize_embeddings: Overrides `normalize_default` if provided.
            convert_to_numpy: Return `np.ndarray` (when `convert_to_tensor=False`).
            convert_to_tensor: Return `torch.Tensor` instead of numpy.
            truncate_dim: If provided, truncates embedding dimension on the last axis.
            prompt: If provided, prepends to each text (simple string concatenation).
            **kwargs: Extra MTEB arguments (ignored).

        Returns:
            An embedding matrix of shape `(N, D)`.

        Raises:
            ValueError: If `pooling` is not set for non-`StaticEmbeddingsModel` models.
            TypeError: If the batch/text format is not supported.
        """
        texts = self._collect_texts(inputs)

        # MTEB может вызвать encode на пустом наборе
        if not texts:
            dim = self.get_sentence_embedding_dimension()
            empty = torch.empty((0, dim), dtype=torch.float32)
            return empty if convert_to_tensor else empty.numpy()

        if prompt:
            texts = [prompt + text for text in texts]

        normalize = (
            self.normalize_default
            if normalize_embeddings is None
            else normalize_embeddings
        )

        all_embeddings: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            if isinstance(self.model, StaticEmbeddingsModel):
                if self.special_tokens.device != inputs["input_ids"].device:
                    self.special_tokens = self.special_tokens.to(
                        inputs["input_ids"].device
                    )
                spec_tok_mask = torch.isin(
                    inputs["input_ids"],
                    self.special_tokens.to(inputs["input_ids"].device),
                    invert=True,
                )
                inputs["attention_mask"] = (
                    inputs["attention_mask"] * spec_tok_mask
                ).long()

            output: LednikModelOutput | StaticEmbeddingsOutput = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            emb = output.sentence_embeddings

            if truncate_dim is not None:
                emb = emb[:, :truncate_dim]

            if normalize:
                emb = F.normalize(emb, p=2, dim=1)

            all_embeddings.append(emb.detach().float().cpu())

        embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_tensor:
            return embeddings
        if convert_to_numpy:
            return embeddings.numpy()
        return embeddings.numpy()

    def similarity(
        self, embeddings1: np.ndarray | Tensor, embeddings2: np.ndarray | Tensor
    ) -> Tensor:
        """Compute a cosine-similarity matrix between two embedding sets.

        Inputs are converted to `torch.Tensor`, moved to the model device and L2-normalized.

        Args:
            embeddings1: Embeddings of shape `(N, D)`.
            embeddings2: Embeddings of shape `(M, D)`.

        Returns:
            A tensor of shape `(N, M)` with cosine similarities.
        """
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        a = embeddings1.to(self.device, dtype=torch.float32)
        b = embeddings2.to(self.device, dtype=torch.float32)
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
        return a @ b.T

    def similarity_pairwise(
        self, embeddings1: np.ndarray | Tensor, embeddings2: np.ndarray | Tensor
    ) -> Tensor:
        """Compute pairwise cosine similarity for two embedding tensors.

        Assumes `embeddings1[i]` is compared with `embeddings2[i]`. Inputs are converted
        to `torch.Tensor`, moved to the model device and L2-normalized.

        Args:
            embeddings1: Embeddings of shape `(N, D)`.
            embeddings2: Embeddings of shape `(N, D)`.

        Returns:
            A tensor of shape `(N,)` with pairwise cosine similarities.
        """
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        a = embeddings1.to(self.device, dtype=torch.float32)
        b = embeddings2.to(self.device, dtype=torch.float32)
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
        return (a * b).sum(dim=-1)

    def get_sentence_embedding_dimension(self) -> int:
        """Return the sentence embedding dimension.

        Uses `model.config.output_hidden_size` (if set) or falls back to `hidden_size`.
        """
        config = self.model.config
        return int(config.output_hidden_size or config.hidden_size)

    def _collect_texts(
        self, inputs: str | list[str] | DataLoader[BatchedInput]
    ) -> list[str]:
        if isinstance(inputs, str):
            return [inputs]

        if isinstance(inputs, list) and (not inputs or isinstance(inputs[0], str)):
            return inputs  # type: ignore[return-value]  # ty:ignore[invalid-return-type]

        texts: list[str] = []

        # MTEB 2.x: DataLoader yielding {"text": list[str]} или похожие batch dict'и
        for batch in inputs:
            if isinstance(batch, dict):
                value = batch.get("text")

                if isinstance(value, str):
                    texts.append(value)
                elif isinstance(value, list):
                    texts.extend(value)
                elif isinstance(value, tuple):
                    texts.extend(list(value))  # ty:ignore[invalid-argument-type]
                else:
                    raise TypeError(f"Unsupported batch['text'] type: {type(value)}")
            elif isinstance(batch, str):
                texts.append(batch)
            elif isinstance(batch, list):
                texts.extend(batch)
            else:
                raise TypeError(f"Unsupported MTEB batch type: {type(batch)}")

        return texts

    @property
    def model_name(self) -> str:
        """Model name for logging."""
        name = self.model_name_for_meta or type(self.model).__name__
        return "Lednik/" + name.replace(" ", "")

    @property
    def mteb_model_meta(self) -> ModelMeta:
        """Metadata of the model."""

        def _get_self(*args: Any, **kwargs: Any) -> "MTEBModelWrapper":
            return self

        return ModelMeta(
            loader=None,
            name=self.model_name,
            revision=self.model_id,
            release_date=None,
            languages=["rus"],
            n_parameters=sum(p.numel() for p in self.model.parameters()),
            memory_usage_mb=None,
            max_tokens=self.max_length,
            embed_dim=self.get_sentence_embedding_dimension(),
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=True,
            framework=["PyTorch"],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )
