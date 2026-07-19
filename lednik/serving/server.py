from tokenizers import Tokenizer
import torch
from transformers import AutoModel, AutoTokenizer
from lednik.models import (
    is_lednik_checkpoint,
    AutoLednikModel,
    LednikPreTrainedModel,
    LednikModelOutput,
    StaticEmbeddingsOutput,
)
from lednik.path_utils import determine_path
from litserve.specs.base import LitSpec
from typing import Union, Optional, TYPE_CHECKING, override, cast
import litserve as ls
from litserve import LitAPI
from kostyl.utils import setup_logger
from transformers.modeling_utils import PreTrainedModel
from transformers import SentencePieceBackend, TokenizersBackend
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput as TransformersModelOutput
from pydantic import BaseModel, Field, model_validator
import click

if TYPE_CHECKING:
    from litserve.loops.base import LitLoop
    from litserve.mcp import MCP

logger = setup_logger(fmt="message_only")

type ModelInstance = LednikPreTrainedModel | PreTrainedModel
type TokenizerInstance = SentencePieceBackend | TokenizersBackend
type ModelOutput = LednikModelOutput | StaticEmbeddingsOutput | TransformersModelOutput


class EmbedRequest(BaseModel):
    """Temporary tokens-only request payload (bypasses the OpenAI spec for load tests)."""

    token_ids: list[int] | None = Field(
        default=None,
        validate_default=False,
        min_length=1,
    )
    text: str | None = Field(
        default=None,
        validate_default=False,
        min_length=1,
    )

    @model_validator(mode="before")
    @classmethod
    def _check_token_ids_or_text(cls, values: dict) -> dict:
        # non-dict bodies fall through to pydantic's own validation (422, not 500)
        if isinstance(values, dict) and (values.get("token_ids") is None) == (
            values.get("text") is None
        ):
            raise ValueError(
                "Request must contain either 'token_ids' or 'text', but not both."
            )
        return values


class LednikServer(LitAPI):
    """A server class for serving Lednik models using LitServe."""

    def __init__(
        self,
        model_path_name_or_id: str,
        tokenizer_path_name_or_id: str,
        max_seq_length: int | None = None,
        max_batch_size: int = 1,
        batch_timeout: float = 0.0,
        api_path: str = "/predict",
        stream: bool = False,
        loop: Union[str, "LitLoop"] | None = "auto",
        spec: LitSpec | None = None,
        mcp: Optional["MCP"] = None,
        enable_async: bool = False,
    ) -> None:
        """
        Initialize LitAPI with configuration options.

        Args:
            model_path_name_or_id: Local path, ClearML model ID or HF Hub repo id of the model checkpoint.
            tokenizer_path_name_or_id: Local path, ClearML model ID or HF Hub repo id of the tokenizer.
            max_seq_length: Maximum sequence length for tokenization. Defaults to None (use model's max).
            max_batch_size: Batch multiple requests for better GPU utilization. Defaults to 1.
            batch_timeout: Wait time for batch to fill (seconds). Defaults to 0.0.
            stream: Enable streaming responses for real-time output. Defaults to False.
            api_path: URL endpoint path. Defaults to "/predict".
            enable_async: Enable async/await for non-blocking operations. Defaults to False.
            spec: API specification (e.g., OpenAISpec for OpenAI compatibility). Defaults to None.
            mcp: Model Context Protocol integration for AI assistants. Defaults to None.
            loop: Event loop for async operations. Defaults to "auto" (auto-detect).
        """
        super().__init__(
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
            api_path=api_path,
            stream=stream,
            loop=loop,
            spec=spec,
            mcp=mcp,
            enable_async=enable_async,
        )

        self.model_path = determine_path(
            model_path_name_or_id,
            is_tokenizer=False,
        )
        self.tokenizer_path = determine_path(
            tokenizer_path_name_or_id,
            is_tokenizer=True,
        )

        self.max_seq_length = max_seq_length
        self.model: ModelInstance | None = None
        self.tokenizer: TokenizerInstance | None = None
        self.backend_tokenizer: Tokenizer | None = None
        return

    @override
    def setup(self, device: str | torch.device) -> None:
        # Device and dtype setup
        if isinstance(device, str):
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(
                f"Device '{device}' is not a CUDA device. Only CUDA devices are supported."
            )
        torch.set_default_device(device)
        torch.cuda.set_device(device)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(
            f"Using device: {device} ({torch.cuda.get_device_name(device)}) with dtype: {dtype}."
        )

        # model loading
        if is_lednik_checkpoint(self.model_path):
            self.model: LednikPreTrainedModel = AutoLednikModel.from_pretrained(
                self.model_path,
                weights_prefix="student.",
                strict_prefix=True,
            )
        else:
            self.model: PreTrainedModel = AutoModel.from_pretrained(self.model_path)

        # tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if tokenizer is None:
            raise ValueError(
                f"Failed to load tokenizer from path '{self.tokenizer_path}'."
            )

        # tokenizer  and model setup
        self.tokenizer = tokenizer
        self.model.to(device=device, dtype=dtype).eval()  # ty:ignore[missing-argument]
        logger.info("Model and tokenizer loaded successfully.")

        # Max sequence length setup
        # determine values
        max_model_length = getattr(
            self.model.config,
            "max_position_embeddings",
            int(self.tokenizer.model_max_length),
        )
        max_seq_length = self.max_seq_length or max_model_length
        # adjust if needed
        if max_seq_length % 8 != 0:
            max_seq_length = (max_seq_length // 8) * 8
            logger.warning(
                f"Adjusted max_seq_length to {max_seq_length} to be a multiple of 8 for better GPU performance."
            )
        self.max_seq_length = max_seq_length
        logger.info(f"Max sequence length set to {self.max_seq_length}.")

        # Backend tokenizer setup
        self.backend_tokenizer = self.tokenizer.backend_tokenizer
        if not isinstance(self.backend_tokenizer, Tokenizer):
            raise TypeError(
                f"Expected backend tokenizer to be of type 'Tokenizer', but got '{type(self.backend_tokenizer).__name__}'."
            )
        self.backend_tokenizer.no_padding()
        self.backend_tokenizer.enable_truncation(max_length=self.max_seq_length)
        return

    @override
    def decode_request(self, request: EmbedRequest) -> list[int] | str:  # ty:ignore[invalid-method-override]
        """Extract the token sequence from the payload FastAPI already validated."""
        if request.token_ids is not None:
            result = request.token_ids
        elif request.text is not None:
            result = request.text
        else:
            raise ValueError("Request must contain either 'token_ids' or 'text'.")
        return result

    def _tokenize_texts(self, texts: list[str]) -> list[torch.Tensor]:
        """Tokenize a list of texts into token ID sequences."""
        if self.backend_tokenizer is None:
            raise RuntimeError(
                "Backend tokenizer must be set up before calling _tokenize_texts."
            )
        encoded = self.backend_tokenizer.encode_batch_fast(texts)
        token_ids_list = [torch.tensor(enc.ids, dtype=torch.long) for enc in encoded]
        return token_ids_list

    @override
    def batch(self, inputs: list[list[int] | str]) -> dict[str, torch.Tensor]:
        """Truncate and pad the requests' token sequences, build the attention mask."""
        if (
            self.model is None
            or self.tokenizer is None
            or self.backend_tokenizer is None
        ):
            raise RuntimeError(
                "Model, tokenizer, and max_seq_length must be set up before calling batch."
            )

        texts: list[str] = []
        text_indices: list[int] = []
        tokens: list[torch.Tensor | None] = []
        for i, input_seq in enumerate(inputs):
            if isinstance(input_seq, str):
                texts.append(input_seq)
                text_indices.append(i)
                tokens.append(None)  # Placeholder for later tokenization
            else:
                truncated_seq = input_seq[: self.max_seq_length]
                tokens.append(torch.tensor(truncated_seq, dtype=torch.long))

        if len(texts) > 0:
            tokenized_sequences = self._tokenize_texts(texts)
            for idx, tokenized_seq in zip(
                text_indices, tokenized_sequences, strict=True
            ):
                truncated_seq = tokenized_seq[: self.max_seq_length]
                tokens[idx] = truncated_seq

        max_seq_length_in_batch = max(len(t) for t in tokens if t is not None)
        # Round up to nearest multiple of 8
        pad_len = (-max_seq_length_in_batch % 8) + max_seq_length_in_batch

        input_ids_buff = torch.full(
            (len(inputs), pad_len),
            fill_value=int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device="cpu",
        )
        attention_mask_buff = torch.zeros(
            (len(inputs), pad_len),
            dtype=torch.long,
            device="cpu",
        )
        for i, token_seq in enumerate(tokens):
            if token_seq is None:
                raise RuntimeError(f"Token sequence at index {i} is None.")
            input_ids_buff[i, : len(token_seq)] = token_seq
            attention_mask_buff[i, : len(token_seq)] = 1

        input_ids = input_ids_buff
        attention_mask = attention_mask_buff
        return {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
        }

    @override
    @torch.inference_mode()
    def predict(  # ty:ignore[invalid-method-override]
        self,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run inference on a batch collated by `batch()`."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model and tokenizer must be set up before calling predict."
            )

        outputs: ModelOutput = self.model(**inputs)

        if isinstance(outputs, LednikModelOutput | StaticEmbeddingsOutput):
            sentence_embeddings = outputs.sentence_embeddings
        elif hasattr(outputs, "pooler_output"):
            sentence_embeddings = cast(torch.Tensor, outputs.pooler_output)
        else:  # Fallback to mean pooling of the last hidden state
            logger.warning_once(
                "Model output does not have 'sentence_embeddings' or 'pooler_output'. "
                "Falling back to mean pooling of the last hidden state."
            )

            last_hidden_state: torch.Tensor | None = getattr(
                outputs, "last_hidden_state", None
            )
            if last_hidden_state is None:
                raise ValueError(
                    "Model output does not contain 'last_hidden_state'. Cannot perform mean pooling."
                )
            elif last_hidden_state.ndim != 3:
                raise ValueError(
                    f"Expected 'last_hidden_state' to be 3D (batch_size, seq_length, hidden_size), "
                    f"but got {last_hidden_state.ndim}D."
                )

            attention_mask: torch.Tensor = inputs.get(
                "attention_mask",
                torch.ones(
                    last_hidden_state.shape[:-1],
                    device=last_hidden_state.device,
                    dtype=torch.long,
                ),
            )
            sentence_embeddings = last_hidden_state.sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)
        return sentence_embeddings.float().cpu()

    @override
    def encode_response(self, output: torch.Tensor) -> dict[str, list[float]]:  # ty:ignore[invalid-method-override]
        """Serialize one embedding; the batched loop passes one row per request."""
        if output.ndim == 2:
            # the non-batched loop skips unbatch and passes [1, D]
            output = output.squeeze(0)
        return {"embedding": output.tolist()}


@click.command()
@click.option(
    "--model",
    "model_path_name_or_id",
    type=str,
    required=True,
    help="Local path, ClearML model ID or HF Hub repo id of the model checkpoint.",
)
@click.option(
    "--tokenizer",
    "tokenizer_path_name_or_id",
    type=str,
    required=True,
    help="Local path, ClearML model ID or HF Hub repo id of the tokenizer.",
)
@click.option(
    "--devices",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of CUDA devices to serve on.",
)
@click.option(
    "--num-workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of worker processes to spawn for serving.",
)
@click.option(
    "--max-seq-length",
    type=int,
    default=None,
    show_default=True,
    help="Truncation length for tokenization; defaults to the model's maximum.",
)
@click.option(
    "--max-batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Maximum number of requests batched together for a single forward pass.",
)
@click.option(
    "--batch-timeout",
    type=float,
    default=0.0,
    show_default=True,
    help="Seconds to wait while collecting a batch before running inference.",
)
@click.option(
    "--fast-queue",
    type=bool,
    default=False,
    show_default=True,
    help="Use ZMQ transport between API servers and workers instead of multiprocessing manager queues.",
)
def run(
    model_path_name_or_id: str,
    tokenizer_path_name_or_id: str,
    devices: int,
    num_workers: int = 1,
    max_seq_length: int | None = None,
    max_batch_size: int = 1,
    batch_timeout: float = 0.0,
    fast_queue: bool = False,
) -> None:
    """Serve a Lednik or Transformers embedding model via LitServe."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please run on a machine with CUDA support."
        )
    if devices > torch.cuda.device_count():
        raise ValueError(
            f"Requested {devices} devices, but only {torch.cuda.device_count()} CUDA devices are available."
        )

    api = LednikServer(
        model_path_name_or_id=model_path_name_or_id,
        tokenizer_path_name_or_id=tokenizer_path_name_or_id,
        max_seq_length=max_seq_length,
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
    )
    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=devices,
        workers_per_device=num_workers,
        fast_queue=fast_queue,
    )
    server.run()
    return


if __name__ == "__main__":
    run()
