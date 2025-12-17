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

from lednik.static_embeddings import StaticEmbeddingsModel


logger = setup_logger(fmt="only_message", add_rank=False)


class StaticEmbeddingsServer(ls.LitAPI):  # noqa: D101
    def __init__(
        self,
        model_path: str | Path,
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
        self.model_path = model_path
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
        self.model = StaticEmbeddingsModel.from_pretrained(self.model_path)
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
def main(model_id: str, port) -> None:
    """Download model from ClearML and print the local path."""
    input_model = InputModel(model_id=model_id)
    local_path = input_model.get_local_copy(raise_on_error=True)
    api = StaticEmbeddingsServer(model_path=local_path, spec=ls.OpenAIEmbeddingSpec())
    server = ls.LitServer(lit_api=api)
    server.run(port=port, generate_client_file=False, pretty_logs=True)
    return


if __name__ == "__main__":
    main()
