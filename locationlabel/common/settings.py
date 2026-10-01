import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".cache"

load_dotenv(ROOT_DIR / ".env")


def env_path(name: str, default: Path) -> Path:
    """Path from an env var; relative values are resolved against the project root."""
    value = os.getenv(name)
    return ROOT_DIR / value if value else default


def output_dir() -> Path:
    return env_path("OUTPUT_DIR", ROOT_DIR / "output")
