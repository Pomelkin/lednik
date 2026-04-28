from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from kostyl.utils import setup_logger
import datetime

logger = setup_logger(fmt="default")


class Logger(ABC):
    """Base logger that buffers metrics and flushes them by training step."""

    def __init__(self) -> None:
        """Initialize an empty in-memory buffer for metric values."""
        self.metrics_buffer: dict[str, list[float]] = defaultdict(list)
        return

    def add_metric_to_buffer(self, metric_name: str, value: float) -> None:
        """Append a metric value to the buffer under the given metric name."""
        self.metrics_buffer[metric_name].append(value)
        return

    def log_metrics(self, step: int) -> None:
        """Persist buffered metrics for a step and clear the buffer afterwards."""
        try:
            self._log_metrics_impl(step=step)
        except Exception as e:
            logger.error(f"Error occurred while logging metrics: {e}")
        self.clear_buffer()
        return

    @abstractmethod
    def _log_metrics_impl(self, step: int) -> None:
        raise NotImplementedError(
            "_log_metrics_impl must be implemented by subclasses of Logger"
        )

    def clear_buffer(self) -> None:
        """Reset the metrics buffer and drop all accumulated values."""
        self.metrics_buffer = defaultdict(list)
        return


class JsonlLogger(Logger):
    """Logger implementation that writes aggregated metrics into a JSONL file."""

    def __init__(self, output_path: str | Path) -> None:
        """Create a JSONL logger and validate or prepare the output path."""
        super().__init__()
        if isinstance(output_path, str):
            output_path = Path(output_path)
        self.output_path = output_path
        self._check_and_setup()
        return

    def _check_and_setup(self) -> None:
        if self.output_path.suffix != ".jsonl":
            if self.output_path.suffix == "":
                if self.output_path.is_file():
                    raise ValueError(
                        f"Output file {self.output_path} does not have an extension"
                    )
                logger.warning(
                    f"Output path {self.output_path} does not have an extension."
                    "Assuming that given path is directory and appending 'metrics.jsonl' to it."
                )
                self.output_path = self.output_path / "metrics.jsonl"
            else:
                raise ValueError(
                    f"Output path must have .jsonl extension, got {self.output_path.suffix}"
                )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists():
            self.output_path.touch()
        else:
            if self.output_path.stat().st_size > 0:
                logger.warning(
                    f"Output file {self.output_path} already exists and is not empty. Metrics will be appended to this file."
                )
        return

    def _log_metrics_impl(self, step: int) -> None:
        metrics2log: dict[str, float | str] = {
            "datetime": datetime.datetime.now().isoformat(),
            "step": step,
        }
        for metric_name, values in self.metrics_buffer.items():
            metrics2log[metric_name] = sum(values) / (len(values) or 1)

        with self.output_path.open("a") as f:
            f.write(f"{metrics2log}\n")
        return
