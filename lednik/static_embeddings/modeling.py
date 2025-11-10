import os
from collections.abc import Callable
from typing import Any
from typing import cast
from typing import override
from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging

from .outputs import StaticEmbeddingsOutput
from .outputs import StaticEmbeddingsPostTrainingOutput
from .outputs import StaticEmbeddingsSequenceClassifierOutput
from lednik.static_embeddings.config import StaticEmbeddingsConfig

logger = logging.get_logger(__name__)

TPreTrainedModel = TypeVar("TPreTrainedModel", bound="PreTrainedModel")


class StaticEmbeddingsPreTrainedModel(PreTrainedModel):
    tokenizer: PreTrainedTokenizerBase | None
    config: StaticEmbeddingsConfig
    base_model_prefix = "model"

    @override
    def _init_weights(self, module: nn.Module) -> None:
        match module:
            case nn.Embedding() if module.weight.dtype != torch.int8:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            case nn.LayerNorm():
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            case nn.RMSNorm():
                nn.init.ones_(module.weight)
            case nn.Linear():
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            case _:
                pass
        return

    def add_tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Add tokenizer to the model."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    @override
    @classmethod
    def from_pretrained(
        cls: type[TPreTrainedModel],
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        load_tokenizer: bool = True,
        **kwargs: dict[str, Any],
    ) -> TPreTrainedModel:
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                use_fast=True,
                revision=revision,
                local_files_only=local_files_only,
                token=token,
            )
            model.add_tokenizer(tokenizer)
        return model

    def save_pretrained(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        save_directory: str | os.PathLike,
        is_main_process: bool = True,
        state_dict: dict | None = None,
        save_function: Callable = torch.save,
        max_shard_size: int | str = "5GB",
        safe_serialization: bool = True,
        variant: str | None = None,
    ) -> None:
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the [`~PreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
                without CPU OOM issues.
                <Tip warning={true}>
                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.
                </Tip>
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.

        """
        super().save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
        )
        if hasattr(self, "tokenizer"):
            if self.tokenizer is not None and is_main_process:
                self.tokenizer.save_pretrained(save_directory)  # type: ignore
        elif (model := getattr(self, self.base_model_prefix, None)) is not None:
            if hasattr(model, "tokenizer"):
                if model.tokenizer is not None and is_main_process:
                    model.tokenizer.save_pretrained(save_directory)  # type: ignore
        return


class StaticEmbeddingsModelForPostTraining(StaticEmbeddingsPreTrainedModel):
    """Static Embeddings for post-training only."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize model."""
        super().__init__(config)
        self.embeddings = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id
        )
        if config.embedding_dropout > 0.0:
            self.dropout = nn.Dropout(config.embedding_dropout)
        else:
            self.dropout = nn.Identity()
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()
        return

    @classmethod
    def initialize(
        cls: type["StaticEmbeddingsModelForPostTraining"],
        config: StaticEmbeddingsConfig,
        embeddings: torch.Tensor,
    ) -> "StaticEmbeddingsModelForPostTraining":
        """Initialize model with given embeddings."""
        model = cls(config)
        model.update_embeddings(embeddings)
        return model

    @override
    @classmethod
    def from_pretrained(
        cls: type["StaticEmbeddingsModelForPostTraining"],
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        load_tokenizer: bool = True,
        **kwargs: dict[str, Any],
    ) -> "StaticEmbeddingsModelForPostTraining":
        if load_tokenizer:
            raise ValueError(
                "Loading tokenizer is not supported for StaticEmbeddingsModelForPostTraining."
            )
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            load_tokenizer=load_tokenizer,
            **kwargs,
        )
        return model

    def update_embeddings(self, new_embeddings: torch.Tensor) -> None:
        """Replace the current model embeddings weights with given one."""
        self.embeddings.weight.data = new_embeddings.clone()
        return

    def to_static_model(
        self, token_pos_weights: torch.Tensor, tokenizer: PreTrainedTokenizerBase
    ) -> "StaticEmbeddingsModel":
        """Convert PostTraining model to StaticEmbeddingsModel."""
        original_device = self.device

        self.cpu()
        token_pos_weights = token_pos_weights.to("cpu")
        with torch.device("cpu"):
            cfg = self.config
            static_model = StaticEmbeddingsModel(cfg)

            embeddings_weights = self.embeddings.weight.data.clone()
            static_model.embeddings.weight.data = embeddings_weights

            static_model.token_pos_weights.data = token_pos_weights.clone()
            static_model.add_tokenizer(tokenizer)
        return static_model.to(original_device)  # type: ignore

    def forward(self, input_ids: torch.Tensor) -> StaticEmbeddingsPostTrainingOutput:
        """Forward pass."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        embeddings = self.embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        output = StaticEmbeddingsPostTrainingOutput(embeddings=embeddings)
        return output


class StaticEmbeddingsModel(StaticEmbeddingsPreTrainedModel):
    """Static Embeddings Model."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize model."""
        super().__init__(config)
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        self.token_pos_weights = nn.Parameter(torch.ones(config.vocab_size, 1))
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase | None = None

        # Initialize weights and apply final processing
        self.post_init()
        return

    def add_tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Add tokenizer to the model."""
        self.tokenizer = tokenizer
        return

    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = -1,
        add_progress_bar: bool = False,
    ) -> list[StaticEmbeddingsOutput]:
        """
        Encode texts into static embeddings.

        Args:
            texts: A list of input texts or a single text string.
            batch_size: The number of texts to process in each batch. If -1, process all texts at once.
            add_progress_bar: Whether to display a progress bar during encoding.

        Returns:
            A list of StaticEmbeddingsOutput containing embeddings, position weights, and sentence embeddings for each text.

        """
        if isinstance(texts, str):
            texts = [texts]
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer is not set for the model. Call `add_tokenizer` method first."
            )

        input_tokens_list = []
        for text in texts:
            encoding = self.tokenizer(
                text, return_tensors="pt", return_attention_mask=False
            )
            encoding = cast(dict[str, torch.Tensor], encoding)
            input_tokens_list.append(encoding["input_ids"].squeeze(0))
        input_tokens = torch.nn.utils.rnn.pad_sequence(
            input_tokens_list,
            batch_first=True,
            padding_value=self.config.pad_token_id,
        ).to(self.device)

        batch_size = batch_size if batch_size > 0 else len(input_tokens)
        outputs = []
        pg = (
            tqdm(total=len(input_tokens), desc="Encoding", unit="text")
            if add_progress_bar
            else None
        )
        for i in range(0, len(input_tokens), batch_size):
            input_ids = input_tokens[i : i + batch_size]
            mask = (input_ids != self.config.pad_token_id).long()
            batch_output: StaticEmbeddingsOutput = self(input_ids, mask)
            for j in range(input_ids.size(0)):
                outputs.append(
                    StaticEmbeddingsOutput(
                        embeddings=batch_output.embeddings[j],
                        pos_weights=batch_output.pos_weights[j],
                        sentence_embeddings=batch_output.sentence_embeddings[j],
                    )
                )
            if pg is not None:
                pg.update(input_ids.size(0))
        if pg is not None:
            pg.close()
        return outputs

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor
    ) -> StaticEmbeddingsOutput:
        """Forward pass."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        embeddings = self.embeddings(input_ids)

        token_weights = self.token_pos_weights[input_ids]  # type: ignore
        masked_embeddings = embeddings * mask.unsqueeze(-1)
        sentence_embeddings = (embeddings * token_weights).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        )
        output = StaticEmbeddingsOutput(
            embeddings=masked_embeddings,
            pos_weights=token_weights,
            sentence_embeddings=sentence_embeddings,
        )
        return output


class StaticEmbeddingsClassfier(nn.Module):
    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize classifier."""
        super().__init__()
        self.dropout = (
            nn.Dropout(config.classifier_dropout)
            if config.classifier_dropout > 0.0
            else nn.Identity()
        )
        self.norm = nn.RMSNorm(config.embedding_dim)
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)
        return

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.dropout(self.norm(sentence_embeddings)))
        logits = self.classifier(x)
        return logits


class StaticEmbeddingsForSequenceClassification(StaticEmbeddingsPreTrainedModel):
    """Static Embeddings Model for sequence classification."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize model."""
        super().__init__(config)
        self.model = StaticEmbeddingsModel(config)
        self.classifier = StaticEmbeddingsClassfier(config)

        # Initialize weights and apply final processing
        self.post_init()
        return

    def add_tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Add tokenizer to the model."""
        self.model.add_tokenizer(tokenizer)
        return

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor
    ) -> StaticEmbeddingsSequenceClassifierOutput:
        """Forward pass."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        embeddings_output = self.model(input_ids, mask)
        logits = self.classifier(embeddings_output.sentence_embeddings)
        output = StaticEmbeddingsSequenceClassifierOutput(
            logits=logits,
            pos_weights=embeddings_output.pos_weights,
            embeddings=embeddings_output.embeddings,
            sentence_embeddings=embeddings_output.sentence_embeddings,
        )
        return output

    def classify(
        self,
        texts: list[str] | str,
        batch_size: int = -1,
        add_progress_bar: bool = False,
    ) -> list[StaticEmbeddingsSequenceClassifierOutput]:
        """
        Classify texts into static embeddings.

        Args:
            texts: A list of input texts or a single text string.
            batch_size: The number of texts to process in each batch. If -1, process all texts at once.
            add_progress_bar: Whether to display a progress bar during classification.

        Returns:
            A list of StaticEmbeddingsSequenceClassifierOutput containing logits, position weights, embeddings, and sentence embeddings for each text.

        """
        if isinstance(texts, str):
            texts = [texts]
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer is not set for the model. Call `add_tokenizer` method first."
            )

        input_tokens_list = []
        for text in texts:
            encoding = self.tokenizer(
                text, return_tensors="pt", return_attention_mask=False
            )
            encoding = cast(dict[str, torch.Tensor], encoding)
            input_tokens_list.append(encoding["input_ids"].squeeze(0))
        input_tokens = torch.nn.utils.rnn.pad_sequence(
            input_tokens_list,
            batch_first=True,
            padding_value=self.config.pad_token_id,
        ).to(self.device)

        batch_size = batch_size if batch_size > 0 else len(input_tokens)
        outputs: list[StaticEmbeddingsSequenceClassifierOutput] = []
        pg = (
            tqdm(total=len(input_tokens), desc="Classifying", unit="text")
            if add_progress_bar
            else None
        )
        for i in range(0, len(input_tokens), batch_size):
            input_ids = input_tokens[i : i + batch_size]
            mask = (input_ids != self.config.pad_token_id).long()
            batch_output: StaticEmbeddingsSequenceClassifierOutput = self(
                input_ids, mask
            )
            for j in range(input_ids.size(0)):
                outputs.append(
                    StaticEmbeddingsSequenceClassifierOutput(
                        logits=batch_output.logits[j],
                        pos_weights=batch_output.pos_weights[j],
                        embeddings=batch_output.embeddings[j],
                        sentence_embeddings=batch_output.sentence_embeddings[j],
                    )
                )
            if pg is not None:
                pg.update(input_ids.size(0))
        if pg is not None:
            pg.close()
        return outputs
