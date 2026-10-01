"""Looks for a newer release by reading a small manifest (latest.txt) from the download server."""

import logging
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .version import APP_VERSION

logger = logging.getLogger(__name__)


@dataclass
class UpdateInfo:
    version: str
    url: str
    notes: str


def manifest_url() -> str:
    """The installer writes update_url.txt next to the exe; a developer run has none, so no check happens."""
    try:
        return (Path(sys.executable).parent / "update_url.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def parse_manifest(text: str) -> dict[str, str]:
    pairs = (line.split("=", 1) for line in text.splitlines() if "=" in line and not line.lstrip().startswith("#"))
    return {key.strip(): value.strip() for key, value in pairs}


def check_for_update(timeout: float = 6.0) -> UpdateInfo | None:
    """The newer release, or None when up to date, offline, or no update server is configured."""
    url = manifest_url()
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            values = parse_manifest(response.read().decode("utf-8-sig"))
        if parse_version(values["version"]) > parse_version(APP_VERSION):
            return UpdateInfo(values["version"], values.get("installer_url", ""), values.get("notes", ""))
    except Exception:
        logger.info("Update check skipped", exc_info=True)
    return None
