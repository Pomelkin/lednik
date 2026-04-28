from datetime import datetime
from pathlib import Path
from typing import override

import polars as pl
from kostyl.utils import setup_logger

from .utils import DirLock
from .utils import is_empty_dir


logger = setup_logger()


class Callback:
    """Abstract base class for callbacks to be used during the generation process."""

    def setup(self) -> None:
        """Called once before the generation loop starts. Can be used for setup tasks."""
        pass

    def on_step(self, results: list[dict[str, str]], step: int) -> None:
        """Called after each generation step."""
        pass

    def on_end(self, results: list[dict[str, str]]) -> None:
        """Called at the end of the generation process."""
        pass


class CheckpointCallback(Callback):
    """
    Callback for saving intermediate generation results as Parquet checkpoints on disk.

    Persists results every `frequency` steps. When `release_results_on_save` is True,
    clears the in-memory results buffer after each save to prevent OOM errors on large datasets.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        frequency: int,
        release_results_on_save: bool,
        ckpt_name: str | None = None,
    ) -> None:
        """
        Initializes CheckpointCallback.

        Args:
            checkpoint_dir: Directory for storing checkpoint files. Created automatically
                if it does not exist.
            frequency: Save interval in steps. A checkpoint is written every `frequency`
                generation steps.
            release_results_on_save: If True, clears the results buffer after each
                checkpoint save to free RAM.
            ckpt_name: Optional prefix for checkpoint filenames. When None, filenames
                consist of step number and UUID (if needed). Also used to filter
                pre-existing checkpoints in the directory.

        """
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir = checkpoint_dir.resolve()
        logger.info(f"Using checkpoint directory: {checkpoint_dir}")

        self.checkpoint_dir = checkpoint_dir
        self.frequency = frequency
        self.prefix: str | None = None
        self.release_results_on_save = release_results_on_save
        self.ckpt_name = ckpt_name

        self._last_save_step = 0
        return

    @override
    def setup(self) -> None:
        # Ensure directory exists before building DirLock, as DirLock requires an existing path.
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with DirLock(self.checkpoint_dir):
            if not is_empty_dir(self.checkpoint_dir):
                if self.ckpt_name is not None:
                    files = list(
                        self.checkpoint_dir.glob(f"*{self.ckpt_name}*_ckpt.parquet")
                    )
                    if len(files) > 0:
                        self.prefix = datetime.now().strftime("%Y|%m|%d_%H|%M|%S.")
                        logger.warning(
                            f"Found {len(files)} existing checkpoints with name {self.ckpt_name} in checkpoint directory {self.checkpoint_dir}.\n"
                            f"Prefix {self.prefix} will be added to all checkpoints filenames."
                        )
                else:
                    raise ValueError(
                        f"Checkpoint directory {self.checkpoint_dir} is not empty and no checkpoint name provided."
                        " Please provide a unique checkpoint name or clear the directory."
                    )
        return

    @property
    def checkpoint_prefix(self) -> str:
        """Builds the checkpoint filename prefix."""
        prefix = ""
        if self.ckpt_name is not None:
            prefix += f"{self.ckpt_name}_"
        if self.prefix is not None:
            prefix += f"{self.prefix}_"
        return prefix

    @override
    def on_step(self, results: list[dict[str, str]], step: int) -> None:
        if step - self._last_save_step >= self.frequency:
            checkpoint_filename = f"{self.checkpoint_prefix}{step}_ckpt.parquet"
            checkpoint_path = self.checkpoint_dir / checkpoint_filename

            df = pl.DataFrame(results)

            df.write_parquet(checkpoint_path, compression_level=15)
            if self.release_results_on_save:
                results.clear()
            self._last_save_step = step
        return

    def build_checkpoint_dataframe(self, results: list[dict[str, str]]) -> pl.DataFrame:
        """
        Builds a final DataFrame from saved checkpoints and the current in-memory buffer.

        If `release_results_on_save` is False, skips checkpoint files and builds the
        DataFrame directly from `results`. Otherwise, reads all `*_ckpt.parquet` files
        from `checkpoint_dir` and concatenates them with any results not yet flushed to disk.

        Args:
            results: In-memory results buffer that has not been written to disk yet.

        Returns:
            A concatenated DataFrame containing all generated results.

        Raises:
            ValueError: If `release_results_on_save` is True but no checkpoint files
                are found and the results buffer is empty.
        """
        if not self.release_results_on_save:
            return pl.DataFrame(results)

        ckpts_path = sorted(
            self.checkpoint_dir.glob(f"{self.checkpoint_prefix}*_ckpt.parquet")
        )
        logger.info(f"Found {len(ckpts_path)} checkpoints to concatenate.")

        dfs = []
        if len(results) > 0:
            df = pl.DataFrame(results)
            dfs.append(df)

        for ckpt_path in ckpts_path:
            df = pl.read_parquet(ckpt_path)
            dfs.append(df)

        if len(dfs) > 0:
            final_df = pl.concat(dfs)
            return final_df
        else:
            raise ValueError("No valid checkpoints found.")
