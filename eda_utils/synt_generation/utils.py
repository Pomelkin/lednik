import threading
from pathlib import Path

from filelock import FileLock
from filelock import Timeout


class DirLock:
    """
    Combined lock for synchronizing access to a directory.

    This lock uses two layers:
    - a thread-level lock (`threading.Lock`) to coordinate threads in one process;
    - a file-based lock (`filelock.FileLock`) to coordinate multiple processes.

    Acquisition order: thread lock -> file lock.
    Release order: file lock -> thread lock.

    The lock file is `<directory>/.dirlock` and is intentionally kept on disk.
    """

    def __init__(
        self,
        directory: str | Path,
        timeout: float = -1,
    ) -> None:
        r"""
        Initialize a directory lock.

        Args:
            directory: Target directory to guard with the lock.
                Must exist and be a directory.
            timeout: Default timeout (in seconds) used by `acquire` for both thread and
                file locks. Use `-1` to wait indefinitely.
        """
        path = (
            Path(directory).resolve()
            if isinstance(directory, str)
            else directory.resolve()
        )
        if not path.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")

        self._dir = path
        self._timeout = timeout
        self._thread_lock = threading.Lock()
        self._file_lock = FileLock(str(path / ".dirlock"), timeout=timeout)
        return

    def acquire(self, timeout: float | None = None) -> None:
        """Acquire both thread and file locks. Timeout applies to both."""
        t = self._timeout if timeout is None else timeout

        thread_timeout = t if t >= 0 else -1
        if not self._thread_lock.acquire(timeout=thread_timeout):
            raise Timeout(str(self._dir))

        try:
            self._file_lock.acquire(timeout=t)
        except BaseException:
            # If file lock acquisition fails, release the thread lock to avoid deadlock
            self._thread_lock.release()
            raise
        return

    def release(self) -> None:
        """Release both file and thread locks."""
        try:
            self._file_lock.release()
        finally:
            self._thread_lock.release()
        return

    def __enter__(self) -> "DirLock":
        """Enter the runtime context."""
        self.acquire()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Exit the runtime context."""
        self.release()
        return

    @property
    def is_locked(self) -> bool:
        """Check if the directory is currently locked by any process/thread."""
        return self._file_lock.is_locked


def is_empty_dir(directory: str | Path) -> bool:
    """Checks if a directory is empty."""
    if isinstance(directory, str):
        directory = Path(directory)
    return directory.is_dir() and not any(directory.iterdir())
