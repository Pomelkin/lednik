import torch
from transformers import AutoModel, AutoTokenizer
from lednik.models import (
    is_lednik_checkpoint,
    AutoLednikModel,
    LednikPreTrainedModel,
    LednikModelOutput,
    StaticEmbeddingsOutput,
)
from pathlib import Path
from litserve.specs.base import LitSpec
from typing import Union, Optional, TYPE_CHECKING, override, cast
import litserve as ls
from litserve import LitAPI
from kostyl.utils import setup_logger
from huggingface_hub import repo_exists, snapshot_download
from transformers.modeling_utils import PreTrainedModel
from transformers import SentencePieceBackend, TokenizersBackend
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput as TransformersModelOutput
import click

if TYPE_CHECKING:
    from litserve.loops.base import LitLoop
    from litserve.mcp import MCP

try:
    from clearml import InputModel

    CLEAR_ML_AVAILABLE = True
except ImportError:
    CLEAR_ML_AVAILABLE = False

logger = setup_logger(fmt="message_only")

type ModelInstance = LednikPreTrainedModel | PreTrainedModel
type TokenizerInstance = SentencePieceBackend | TokenizersBackend
type ModelOutput = LednikModelOutput | StaticEmbeddingsOutput | TransformersModelOutput


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

        self.model_path = self._determine_path(
            model_path_name_or_id,
            is_tokenizer=False,
        )
        self.tokenizer_path = self._determine_path(
            tokenizer_path_name_or_id,
            is_tokenizer=True,
        )

        self.max_seq_length = max_seq_length
        self.model: ModelInstance | None = None
        self.tokenizer: TokenizerInstance | None = None
        return

    def _determine_path(self, path_name_or_id: str, is_tokenizer: bool) -> str:
        path: Path | None = None
        # Then check if it's a ClearML model ID
        if CLEAR_ML_AVAILABLE:
            model_id = path_name_or_id
            try:
                clearml_model = InputModel(model_id=model_id)
                local_path = clearml_model.get_local_copy()
                path = Path(local_path).resolve()
                logger.info(
                    f"Resolved ClearML model ID '{model_id}' to local path '{path}'."
                )
            except Exception:  # noqa: S110
                pass

        # If neither, try to check if it's a Hugging Face model ID
        if repo_exists(path_name_or_id):
            repo_id = path_name_or_id
            try:
                local_path = snapshot_download(repo_id=repo_id)
                path = Path(local_path).resolve()
                logger.info(
                    f"Resolved Hugging Face model ID '{repo_id}' to local path '{path}'."
                )
            except Exception:  # noqa: S110
                pass

        # Check if the provided string is a local path
        if path is None:
            path = Path(path_name_or_id)
            logger.info(f"Using provided local path '{path}'.")

        if not path.exists():
            raise FileNotFoundError(
                f"Path '{path}' does not exist. Please provide a valid local path, ClearML model ID, or Hugging Face model ID."
            )

        # Path content validation
        # tokenizer validation
        if is_tokenizer:
            if not (path.is_dir() and (path / "tokenizer.json").exists()):
                raise ValueError(f"Path '{path}' is not a valid tokenizer directory.")
        # model validation
        elif not (
            (path.is_dir() and (path / "config.json").exists())
            or (path.is_file() and path.suffix == ".ckpt")
        ):
            raise ValueError(
                f"Path '{path}' is not a valid Transformers checkpoint directory or Lightning .ckpt file."
            )
        return str(path.resolve())

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

        # model and tokenizer setup
        self.tokenizer = tokenizer
        self.model.to(device=device, dtype=dtype).eval()  # ty:ignore[missing-argument]
        logger.info("Model and tokenizer loaded successfully.")

        # Max sequence length setup
        max_model_length = self.max_seq_length or getattr(
            self.model.config,
            "max_position_embeddings",
            int(self.tokenizer.model_max_length),
        )
        max_seq_length = min(self.max_seq_length or max_model_length, max_model_length)
        if max_seq_length % 8 != 0:
            max_seq_length = (max_seq_length // 8) * 8
            logger.warning(
                f"Adjusted max_seq_length to {max_seq_length} to be a multiple of 8 for better GPU performance."
            )
        self.max_seq_length = max_seq_length
        logger.info(f"Max sequence length set to {self.max_seq_length}.")
        return

    @override
    @torch.inference_mode()
    def predict(  # ty:ignore[invalid-method-override]
        self,
        inputs: list[str],
    ) -> torch.Tensor:
        """Run inference on the model with the provided inputs."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model and tokenizer must be set up before calling predict."
            )

        # Tokenization
        tokenized_inputs = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(self.model.device)

        outputs: ModelOutput = self.model(**tokenized_inputs)

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

            attention_mask: torch.Tensor = tokenized_inputs.get(
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
    "--num-api-servers",
    type=click.IntRange(min=1),
    default=None,
    help="Number of HTTP API server processes; defaults to the number of workers.",
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
    num_api_servers: int | None = None,
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
        spec=ls.OpenAIEmbeddingSpec(),
    )
    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=devices,
        workers_per_device=num_workers,
        fast_queue=fast_queue,
    )
    server.run(num_api_servers=num_api_servers)
    return


if __name__ == "__main__":
    run()
