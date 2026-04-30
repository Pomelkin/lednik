import os
from collections.abc import Callable
from typing import Any
from typing import TypeVar
from typing import cast
from typing import override

import torch
import torch.nn.functional as F
from kostyl.ml.integrations.lightning import LightningCheckpointLoaderMixin
from kostyl.utils import setup_logger
from torch import nn
from tqdm.auto import tqdm  # pyrefly: ignore
from transformers import AutoTokenizer
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers import SentencePieceBackend
from transformers import TokenizersBackend

from .configuration_static import StaticEmbeddingsConfig
from .outputs import StaticEmbeddingsOutput
from .outputs import StaticEmbeddingsSequenceClassifierOutput


logger = setup_logger(fmt="only_message")

TPreTrainedModel = TypeVar("TPreTrainedModel", bound="PreTrainedModel")


class StaticEmbeddingsPreTrainedModel(LightningCheckpointLoaderMixin, PreTrainedModel):
    """
    An abstract base class for static embedding models, inheriting from `PreTrainedModel`.

    This class handles the initialization of weights, tokenizer integration, and provides
    overridden methods for loading and saving pretrained models that include tokenizer handling.
    """

    tokenizer: TokenizersBackend | SentencePieceBackend | None
    config: StaticEmbeddingsConfig  # type: ignore
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

    def add_tokenizer(
        self, tokenizer: TokenizersBackend | SentencePieceBackend
    ) -> None:
        """Add tokenizer to the model."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_tokenizer(self) -> TokenizersBackend | SentencePieceBackend:
        """Get tokenizer from the model."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    @override
    @classmethod
    def from_pretrained(  # type: ignore
        cls: type["StaticEmbeddingsPreTrainedModel"],
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
        **kwargs: Any,
    ) -> "StaticEmbeddingsPreTrainedModel":
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
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    use_fast=True,
                    revision=revision,
                    local_files_only=local_files_only,
                    token=token,
                )
                model.add_tokenizer(tokenizer)  # type: ignore
            except (TypeError, FileNotFoundError, OSError) as e:
                logger.warning(
                    "Tokenizer could not be loaded. Make sure the tokenizer files are present."
                )
                logger.debug(f"Tokenizer loading error: {e}")
                pass
        return model

    def save_pretrained(  # type: ignore
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
        tokenizer = getattr(
            self,
            "tokenizer",
            getattr(getattr(self, self.base_model_prefix, None), "tokenizer", None),
        )
        if tokenizer is not None and is_main_process:
            tokenizer.save_pretrained(save_directory)
        return


class StaticEmbeddingsModel(StaticEmbeddingsPreTrainedModel):
    """Static Embeddings Model."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize model."""
        super().__init__(config)
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_pos_weights = nn.Embedding(
            config.vocab_size,
            1,
        )
        self.output_proj = (
            nn.Linear(config.hidden_size, config.output_hidden_size)
            if config.output_hidden_size is not None
            else nn.Identity()
        )
        self.norm = nn.RMSNorm(config.output_hidden_size or config.hidden_size)
        self.dropout = (
            nn.Dropout(config.embedding_dropout)
            if config.embedding_dropout > 0.0
            else nn.Identity()
        )

        self.config = config
        self.tokenizer: TokenizersBackend | SentencePieceBackend | None = None

        # Initialize weights and apply final processing
        self.post_init()
        return

    @override
    def add_tokenizer(
        self, tokenizer: TokenizersBackend | SentencePieceBackend
    ) -> None:
        self.tokenizer = tokenizer
        return

    @override
    def get_tokenizer(self) -> TokenizersBackend | SentencePieceBackend:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set for the model.")
        return self.tokenizer

    def replace_embeddings(self, new_embeddings: torch.Tensor | nn.Parameter) -> None:
        """Replace the current model embeddings weights with given one."""
        if new_embeddings.numel() != self.embeddings.weight.numel():
            raise ValueError(
                f"new_embeddings should have numel {self.embeddings.weight.numel()}, "
                f"but got {new_embeddings.numel()}."
            )
        if new_embeddings.size() != self.embeddings.weight.size():
            raise ValueError(
                f"new_embeddings should have size {self.embeddings.weight.size()}, "
                f"but got {new_embeddings.size()}."
            )

        new_embeddings = new_embeddings.clone()  # use clone() to avoid weight tying
        if isinstance(new_embeddings, torch.Tensor):
            new_embeddings = nn.Parameter(new_embeddings)  # avoiding weight ties
        self.embeddings.weight = new_embeddings
        return

    def replace_pos_weights(self, new_pos_weights: torch.Tensor | nn.Parameter) -> None:
        """Replace the current model position weights with given one."""
        if new_pos_weights.numel() != self.token_pos_weights.weight.numel():
            raise ValueError(
                f"new_pos_weights should have numel {self.token_pos_weights.weight.numel()}, "
                f"but got {new_pos_weights.numel()}."
            )
        if new_pos_weights.size() != self.token_pos_weights.weight.size():
            raise ValueError(
                f"new_pos_weights should have size {self.token_pos_weights.weight.size()}, "
                f"but got {new_pos_weights.size()}."
            )

        new_pos_weights = new_pos_weights.clone()  # use clone() to avoid weight tying
        if isinstance(new_pos_weights, torch.Tensor):
            new_pos_weights = nn.Parameter(new_pos_weights)  # avoiding weight ties
        self.token_pos_weights.weight = new_pos_weights
        return

    @classmethod
    def initialize(
        cls,
        config: StaticEmbeddingsConfig,
        embeddings: torch.Tensor | nn.Parameter,
    ) -> "StaticEmbeddingsModel":
        """Initialize model with given embeddings."""
        model = cls(config)
        model.replace_embeddings(embeddings)
        return model

    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = -1,
        add_progress_bar: bool = False,
    ) -> StaticEmbeddingsOutput:
        """
        Encode texts into static embeddings.

        Args:
            texts: A list of input texts or a single text string.
            batch_size: The number of texts to process in each batch. If -1, process all texts at once.
            add_progress_bar: Whether to display a progress bar during encoding.

        Returns:
            StaticEmbeddingsOutput containing embeddings, position weights, and sentence embeddings for each text.

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

        batch_size = batch_size if batch_size > 0 else len(input_tokens_list)
        outputs = []
        pg = (
            tqdm(total=len(input_tokens_list), desc="Encoding", unit="text")
            if add_progress_bar
            else None
        )
        for i in range(0, len(input_tokens_list), batch_size):
            input_ids_nc = input_tokens_list[i : i + batch_size]

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_nc,
                batch_first=True,
                padding_value=int(self.config.pad_token_id),  # type: ignore
            ).to(self.device)

            mask = (input_ids != int(self.config.pad_token_id)).long()  # type: ignore
            batch_output: StaticEmbeddingsOutput = self(input_ids, mask)
            outputs.append(batch_output)
            if pg is not None:
                pg.update(input_ids.size(0))
        if pg is not None:
            pg.close()

        result = outputs[0]
        for output in outputs[1:]:
            result.token_embeddings = torch.cat(
                (result.token_embeddings, output.token_embeddings), dim=0
            )
            result.pos_weights = torch.cat(  # type: ignore
                (result.pos_weights, output.pos_weights),
                dim=0,
            )
            result.sentence_embeddings = torch.cat(
                (result.sentence_embeddings, output.sentence_embeddings), dim=0
            )
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> StaticEmbeddingsOutput:
        """Forward pass."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        embeddings = self.output_proj(self.dropout(self.embeddings(input_ids)))
        token_weights = self.token_pos_weights(input_ids)
        embeddings = self.norm(embeddings)

        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        sentence_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-9)
        output = StaticEmbeddingsOutput(
            token_embeddings=embeddings,
            pos_weights=token_weights,
            sentence_embeddings=sentence_embeddings,
        )
        return output


class StaticEmbeddingsClassificationHead(nn.Module):
    """Classification head for static embeddings."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize classifier."""
        super().__init__()
        self.dropout = (
            nn.Dropout(config.classifier_dropout)
            if config.classifier_dropout > 0.0
            else nn.Identity()
        )
        self.norm = nn.RMSNorm(config.hidden_size)
        self.hidden_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config
        return

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.dropout(F.relu(self.hidden_proj(self.norm(sentence_embeddings))))
        logits = self.head(x)
        return logits


class StaticEmbeddingsForSequenceClassification(StaticEmbeddingsPreTrainedModel):
    """Static Embeddings Model for sequence classification."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize model."""
        super().__init__(config)
        if not self.config.is_tokenizer_customized:
            raise ValueError(
                "The tokenizer is not customized. Please use "
                "`lednik.distill.initialization.tokenizer_utils.customize_tokenizer` "
                "to customize the tokenizer before initializing the model."
            )

        self.model = StaticEmbeddingsModel(config)
        self.classifier = StaticEmbeddingsClassificationHead(config)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()
        return

    @override
    def add_tokenizer(
        self, tokenizer: TokenizersBackend | SentencePieceBackend
    ) -> None:
        self.model.add_tokenizer(tokenizer)
        return

    @override
    def get_tokenizer(self) -> TokenizersBackend | SentencePieceBackend:
        return self.model.get_tokenizer()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StaticEmbeddingsSequenceClassifierOutput:
        """Forward pass."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        embeddings_output = self.model(input_ids, attention_mask)
        logits = self.classifier(embeddings_output.sentence_embeddings)
        loss = self.loss_function(logits, labels) if labels is not None else None  #  type: ignore
        output = StaticEmbeddingsSequenceClassifierOutput(
            logits=logits,
            loss=loss,  # type: ignore
        )
        return output

    def classify(
        self,
        texts: list[str] | str,
        batch_size: int = -1,
        add_progress_bar: bool = False,
    ) -> StaticEmbeddingsSequenceClassifierOutput:
        """
        Classify texts into static embeddings.

        Args:
            texts: A list of input texts or a single text string.
            batch_size: The number of texts to process in each batch. If -1, process all texts at once.
            add_progress_bar: Whether to display a progress bar during classification.

        Returns:
            A StaticEmbeddingsSequenceClassifierOutput containing logits, position weights, embeddings, and sentence embeddings for each text.

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
            padding_value=int(self.config.pad_token_id),  # type: ignore
        ).to(self.device)

        batch_size = batch_size if batch_size > 0 else len(input_tokens)
        pg = (
            tqdm(total=len(input_tokens), desc="Classifying", unit="text")
            if add_progress_bar
            else None
        )

        outputs: list[StaticEmbeddingsSequenceClassifierOutput] = []
        for i in range(0, len(input_tokens), batch_size):
            input_ids = input_tokens[i : i + batch_size]
            mask = (input_ids != int(self.config.pad_token_id)).long()  # type: ignore
            batch_output: StaticEmbeddingsSequenceClassifierOutput = self(
                input_ids, mask
            )
            outputs.append(batch_output)
            if pg is not None:
                pg.update(input_ids.size(0))
        if pg is not None:
            pg.close()
        result = outputs[0]
        for output in outputs[1:]:
            result.logits = torch.cat((result.logits, output.logits), dim=0)
            if result.loss is not None and output.loss is not None:
                result.loss += output.loss
        return result
