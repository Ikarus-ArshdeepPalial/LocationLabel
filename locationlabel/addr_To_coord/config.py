import os
from dataclasses import dataclass
from pathlib import Path

from common.settings import CACHE_DIR, env_path, output_dir


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    output_dir: Path
    cache_path: Path
    timeout: float
    max_retries: int


def load_settings() -> Settings:
    return Settings(
        google_api_key=os.getenv("GOOGLE_MAPS_API_KEY") or None,
        output_dir=output_dir(),
        cache_path=env_path("GEOCODE_CACHE_PATH", CACHE_DIR / "geocode.sqlite"),
        timeout=float(os.getenv("HTTP_TIMEOUT", "10")),
        max_retries=int(os.getenv("HTTP_MAX_RETRIES", "3")),
    )
