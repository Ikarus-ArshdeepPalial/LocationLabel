import csv
import logging
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

from common.cache import JsonCache
from common.progress import track_addresses

from .config import load_settings
from .geocode import Geocoder
from .record import GeocodeResult, save_record

logger = logging.getLogger(__name__)


def run(addresses: list[str], force: bool = False, summary_name: str | None = None) -> Counter[str]:
    settings = load_settings()
    cache = JsonCache(settings.cache_path)
    try:
        geocoder = Geocoder(settings, cache)
        results = [process(geocoder, address, settings.output_dir, force) for address in track_addresses(addresses, "[1/4] geocode")]
    finally:
        cache.close()

    if summary_name:
        summary_path = settings.output_dir / f"summary_{summary_name}.csv"
        write_summary(results, summary_path)
        logger.info("Summary saved to %s", summary_path)
    return Counter(r.status for r in results)


def process(geocoder: Geocoder, address: str, output_dir: Path, refresh: bool) -> GeocodeResult:
    try:
        result = geocoder.geocode(address, refresh)
    except Exception as exc:
        logger.exception("Unexpected error for %r", address)
        result = GeocodeResult.failure(address, f"Unexpected error: {exc}")

    path = save_record(result, output_dir)
    logger.debug("Saved %s", path)
    if result.status == "ok":
        logger.info("%r -> (%.6f, %.6f) via %s [%s]", address, result.lat, result.lng, result.provider, result.match_quality)
    else:
        logger.error("%r failed: %s", address, result.error)
    return result


def write_summary(results: list[GeocodeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(GeocodeResult)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
