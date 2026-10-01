import os
from dataclasses import dataclass
from pathlib import Path

from common.settings import CACHE_DIR, env_path, output_dir


@dataclass(frozen=True)
class ViewSpec:
    name: str
    heading_offset: float
    pitch: float


MAIN_VIEWS = (
    ViewSpec("house", 0, 0),
    ViewSpec("house_left", -35, 0),
    ViewSpec("house_right", 35, 0),
    ViewSpec("curb", 0, -18),
    ViewSpec("street_left", -90, 0),
    ViewSpec("street_right", 90, 0),
)

NEIGHBOR_VIEWS = (
    ViewSpec("house", 0, 0),
    ViewSpec("curb", 0, -15),
)


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    output_dir: Path
    cache_path: Path
    timeout: float
    max_retries: int
    image_width: int = 640
    image_height: int = 640
    fov: float = 90
    search_radii: tuple[int, ...] = (50, 100, 200)
    neighbor_step_m: float = 15
    side_offset_m: float = 15
    main_views: tuple[ViewSpec, ...] = MAIN_VIEWS
    neighbor_views: tuple[ViewSpec, ...] = NEIGHBOR_VIEWS


def load_settings() -> Settings:
    width, height = os.getenv("STREETVIEW_SIZE", "640x640").lower().split("x")
    return Settings(
        google_api_key=os.getenv("GOOGLE_MAPS_API_KEY") or None,
        output_dir=output_dir(),
        cache_path=env_path("STREETVIEW_CACHE_PATH", CACHE_DIR / "streetview.sqlite"),
        timeout=float(os.getenv("HTTP_TIMEOUT", "10")),
        max_retries=int(os.getenv("HTTP_MAX_RETRIES", "3")),
        image_width=int(width),
        image_height=int(height),
        fov=float(os.getenv("STREETVIEW_FOV", "90")),
    )
