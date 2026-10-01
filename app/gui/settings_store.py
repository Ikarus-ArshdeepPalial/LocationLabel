import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from common.settings import ROOT_DIR
from measure.limits import DEFAULTS

logger = logging.getLogger(__name__)

SETTINGS_PATH = ROOT_DIR / "settings.json"
ALL_TARGETS = ["ramps", "driveways", "signs", "poles", "curbs"]


def default_limits() -> list[dict]:
    return [asdict(rule) for rule in DEFAULTS]


SOURCE_TYPES = {"Phone or camera photo": "phone", "Google Street View screenshot": "google_screenshot"}


@dataclass
class AppSettings:
    use_street_view: bool = False
    api_key: str = ""
    source_type: str = "phone"
    camera_height: float | None = None
    crop_ui: bool = False
    check_updates: bool = True
    targets: list[str] = field(default_factory=lambda: list(ALL_TARGETS))
    bulk_targets: list[str] = field(default_factory=lambda: list(ALL_TARGETS))
    bulk_input: str = ""
    bulk_output: str = ""
    bulk_skip_done: bool = False
    limits: list[dict] = field(default_factory=default_limits)
    output_dir: str = str(Path.home() / "Documents" / "LocationLabel Results")


def load_settings() -> AppSettings:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        known = {f.name for f in fields(AppSettings)}
        settings = AppSettings(**{k: v for k, v in data.items() if k in known})
        settings.targets = [t for t in settings.targets if t in ALL_TARGETS] or list(ALL_TARGETS)
        settings.bulk_targets = [t for t in settings.bulk_targets if t in ALL_TARGETS] or list(ALL_TARGETS)
        return settings
    except (OSError, ValueError, TypeError):
        return AppSettings()


def save_settings(settings: AppSettings) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    except OSError:
        logger.exception("Could not save settings to %s", SETTINGS_PATH)
