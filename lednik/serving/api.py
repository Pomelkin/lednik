from pathlib import Path
from typing import Optional
from typing import Union

import click
import litserve as ls
import torch
from clearml import InputModel
from kostyl.utils.logging import setup_logger
from litserve.loops.base import LitLoop
from litserve.mcp import MCP
from litserve.specs.base import LitSpec
from transformers import AutoTokenizer

from lednik.models import StaticEmbeddingsModel


logger = setup_logger(fmt="only_message")


class StaticEmbeddingsServer(ls.LitAPI):  # noqa: D101
    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path | None = None,
        max_batch_size: int = 1,
        batch_timeout: float = 0.0,
        api_path: str = "/predict",
        stream: bool = False,
        loop: Union[str, "LitLoop"] | None = "auto",
        spec: LitSpec | None = None,
        mcp: Optional["MCP"] = None,
        enable_async: bool = False,
    ) -> None:
        """Initialize LitAPI with configuration options."""
        if isinstance(model_path, str):
            model_path = Path(model_path)
        if isinstance(tokenizer_path, str):
            tokenizer_path = Path(tokenizer_path)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model: StaticEmbeddingsModel | None = None
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

    def setup(self, device: str) -> None:
        """Load the StaticEmbeddingsModel from the specified path."""
        if self.model_path.is_file() and self.model_path.suffix == ".ckpt":
            if self.tokenizer_path is None:
                raise ValueError(
                    "You are trying to load a model from a Lightning checkpoint file. "
                    "For this scenario `--tokenizer-id` must be specified."
                )
            model = StaticEmbeddingsModel.from_lightning_checkpoint(
                self.model_path, weights_prefix="static_model", strict_prefix=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            model.add_tokenizer(tokenizer)
        else:
            model = StaticEmbeddingsModel.from_pretrained(self.model_path)
        self.model = model
        self.model.to(device)  # type: ignore
        self.model.eval()
        return

    @torch.inference_mode()
    def predict(self, inputs: list[str]) -> torch.Tensor:
        """Generate embeddings for the input texts."""
        if self.model is None:
            raise ValueError("Model is not loaded. Call setup() before predict().")
        embeddings = self.model.encode(inputs)
        return embeddings.sentence_embeddings


@click.command()
@click.option("--model-id", type=str)
@click.option("--port", type=int, default=8080)
@click.option("--tokenizer-id", type=str, default="", required=False)
def main(model_id: str, port: int, tokenizer_id: str | None = None) -> None:
    """Download model from ClearML and print the local path."""
    input_model = InputModel(model_id=model_id)
    model_local_path = input_model.get_local_copy(raise_on_error=True)
    if tokenizer_id != "":
        input_tokenizer = InputModel(model_id=tokenizer_id)
        tokenizer_local_path = input_tokenizer.get_local_copy(raise_on_error=True)
    else:
        tokenizer_local_path = None
    api = StaticEmbeddingsServer(
        model_path=model_local_path,
        tokenizer_path=tokenizer_local_path,
        spec=ls.OpenAIEmbeddingSpec(),
    )
    server = ls.LitServer(lit_api=api)
    server.run(port=port, generate_client_file=False, pretty_logs=True)
    return


if __name__ == "__main__":
    main()
