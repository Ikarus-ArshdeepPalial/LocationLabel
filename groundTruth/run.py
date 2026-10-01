import logging
from collections import Counter
from pathlib import Path

from .build import run_build
from .config import GROUND_TRUTH_CSV, Settings
from .discover import cmd_inspect
from .download import download_all
from .evaluate import run_evaluate
from .run_groundtruth import run as run_groundtruth_rows
from .sources import load_sources

logger = logging.getLogger(__name__)


def inspect(only: list[str] | None = None) -> None:
    cmd_inspect(Settings(), only)


def download(force: bool = False) -> dict[str, int]:
    return download_all(Settings(), load_sources(), force)


def build() -> Path:
    return run_build(Settings())


def run_groundtruth(csv_path: Path = GROUND_TRUTH_CSV, split: str | None = None, limit: int | None = None, force: bool = False, dry_run: bool = False) -> Counter[str]:
    return run_groundtruth_rows(csv_path, split, limit, force, dry_run)


def evaluate(csv_path: Path = GROUND_TRUTH_CSV) -> Path:
    return run_evaluate(csv_path)
