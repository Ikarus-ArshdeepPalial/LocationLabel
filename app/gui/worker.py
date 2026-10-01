import logging
import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .bulk import JobFolder, run_bulk
from .pipeline import friendly_error, run_job
from .settings_store import AppSettings
from .updates import check_for_update

logger = logging.getLogger(__name__)


class _SignalHandler(logging.Handler):
    def __init__(self, signal: Signal) -> None:
        super().__init__(logging.INFO)
        self._signal = signal

    def emit(self, record: logging.LogRecord) -> None:
        self._signal.emit(self.format(record))


class UpdateChecker(QThread):
    found = Signal(object)

    def run(self) -> None:
        info = check_for_update()
        if info:
            self.found.emit(info)


class JobWorker(QThread):
    progress = Signal(float, str)
    log = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: Path | None, address: str, settings: AppSettings, output_dir: Path, targets: tuple[str, ...]) -> None:
        super().__init__()
        self._args = (folder, address, settings)
        self._output_dir, self._targets = output_dir, targets

    def run(self) -> None:
        handler = _SignalHandler(self.log)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        try:
            self.succeeded.emit(run_job(*self._args, self.progress.emit, self._output_dir, self._targets))
        except Exception as exc:
            logger.exception("Analysis failed")
            self.failed.emit(friendly_error(exc))
        finally:
            logging.getLogger().removeHandler(handler)


class BulkWorker(QThread):
    progress = Signal(float, str)
    log = Signal(str)
    finished_with = Signal(object)
    failed = Signal(str)

    def __init__(self, jobs: list[JobFolder], targets: tuple[str, ...], settings: AppSettings, out_root: Path, skip_done: bool) -> None:
        super().__init__()
        self._args = (jobs, targets, settings, out_root)
        self._skip_done = skip_done
        self._stop = threading.Event()

    def stop(self) -> None:
        """Finishes the folders being analyzed now, then stops."""
        self._stop.set()

    def run(self) -> None:
        handler = _SignalHandler(self.log)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        try:
            self.finished_with.emit(run_bulk(*self._args, self.progress.emit, self._stop.is_set, self._skip_done))
        except Exception as exc:
            logger.exception("Bulk analysis failed")
            self.failed.emit(friendly_error(exc))
        finally:
            logging.getLogger().removeHandler(handler)
