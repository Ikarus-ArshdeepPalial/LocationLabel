from collections import Counter
from pathlib import Path

from common.progress import track_addresses

from .calibration import build_calibrators
from .config import load_settings
from .importer import ImageImporter


def run(
    jobs: list[tuple[str, Path]],
    camera_height: float | None = None,
    source_type: str | None = None,
    crop_ui: bool = False,
) -> Counter[str]:
    """Import user-supplied images for each (address, images_dir) job in place of Street View fetching."""
    settings = load_settings(source_type, camera_height, crop_ui)
    importer = ImageImporter(settings, build_calibrators(settings.calibrators, settings.defaults.fov_deg))
    folders = dict(jobs)
    return Counter(importer.run(address, folders[address]) for address in track_addresses(list(folders), "[2/4] import"))
