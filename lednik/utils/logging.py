import inspect
import os
import sys
import uuid
from pathlib import Path
from typing import Literal
from typing import TYPE_CHECKING

from loguru import logger as _base_logger


if TYPE_CHECKING:
    from loguru import Logger

try:
    import torch.distributed as dist
except Exception:

    class _Dummy:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_initialized() -> bool:
            return False

    dist = _Dummy()

_DEFAULT_SINK_REMOVED = False
_DEFAULT_FMT = "<level>{level: <8}</level> {time:HH:mm:ss.SSS} [{extra[channel]}] <level>{message}</level>"
_ONLY_MESSAGE_FMT = "<level>{message}</level>"
_PRESETS = {"default": _DEFAULT_FMT, "only_message": _ONLY_MESSAGE_FMT}


def _caller_filename() -> str:
    """Имя файла, из которого вызвали setup_logger()."""
    # Берём первый фрейм выше текущей функции
    frame = inspect.stack()[2] if len(inspect.stack()) > 2 else inspect.stack()[1]
    name = Path(frame.filename).name
    return name


def setup_logger(
    name: str | None = None,
    fmt: Literal["default", "only_message"] | str = "default",
    level: str = "INFO",
    add_rank: bool | None = None,
    sink=sys.stdout,
    colorize: bool = True,
    serialize: bool = False,
) -> Logger:
    """
    Returns a bound logger with its own sink and formatting.
    If name=None, the caller's filename (similar to __file__) is used automatically.

    Format example: "{level} {time:MM-DD HH:mm:ss} [{extra[channel]}] {message}"
    """

    global _DEFAULT_SINK_REMOVED
    if not _DEFAULT_SINK_REMOVED:
        _base_logger.remove()
        _DEFAULT_SINK_REMOVED = True

    # Определяем имя канала
    if name is None:
        base = _caller_filename()
    else:
        base = name

    # Нужен ли префикс ранка
    if add_rank is None:
        try:
            add_rank = dist.is_available() and dist.is_initialized()
        except Exception:
            add_rank = False

    if add_rank:
        rank = int(os.environ.get("RANK", "0"))
        channel = f"rank:{rank} - {base}"
    else:
        channel = base

    if fmt in _PRESETS:
        fmt = _PRESETS[fmt]
    else:
        fmt = str(fmt)

    # Уникальный id для фильтрации записей именно этого логгера
    logger_id = uuid.uuid4().hex

    # Вешаем отдельный sink с собственным форматом и фильтром по extra.logger_id
    _base_logger.add(
        sink,
        level=level,
        format=fmt,
        colorize=colorize,
        serialize=serialize,
        filter=lambda r: r["extra"].get("logger_id") == logger_id,
    )

    # Возвращаем proxy-логгер, в extra которого проставлены наш id и "channel"
    # Используй: log.info("..."), log.error("..."), и т.д.
    return _base_logger.bind(logger_id=logger_id, channel=channel)
