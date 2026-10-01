import os
from pathlib import Path


def app_home() -> Path:
    """Per-user data folder (settings, caches, staged photos); LOCATIONLABEL_HOME overrides it."""
    override = os.getenv("LOCATIONLABEL_HOME")
    base = Path(override) if override else Path(os.getenv("LOCALAPPDATA") or Path.home()) / "LocationLabel"
    base.mkdir(parents=True, exist_ok=True)
    return base


ROOT_DIR = app_home()
CACHE_DIR = ROOT_DIR / ".cache"


def env_path(name: str, default: Path) -> Path:
    """Path from an env var; relative values are resolved against the app data folder."""
    value = os.getenv(name)
    return ROOT_DIR / value if value else default


def output_dir() -> Path:
    return env_path("OUTPUT_DIR", ROOT_DIR / "output")
