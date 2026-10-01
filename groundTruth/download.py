import json
import logging
import time
from pathlib import Path

import requests

from common.http import build_session
from common.progress import progress

from .arcgis import ArcGISError, layer_info, query_page
from .config import NORMALIZED_DIR, RAW_DIR, Settings
from .normalize import NormalizedRow, normalize_features
from .sources import Source, usable_sources

logger = logging.getLogger(__name__)


def fetch_pages(session: requests.Session, source: Source, settings: Settings, force: bool = False) -> list[dict]:
    """All raw features for a source, caching each page under data/raw/<source>/page_NNNN.json."""
    cache_dir = RAW_DIR / source.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    info = layer_info(session, source.url, settings.http_timeout)
    if not info.reachable:
        raise ArcGISError(info.error or "layer unreachable")
    size = info.max_record_count or 1000

    features: list[dict] = []
    offset, page = 0, 0
    while True:
        page_path = cache_dir / f"page_{page:04d}.json"
        if page_path.exists() and not force:
            data = json.loads(page_path.read_text(encoding="utf-8"))
        else:
            data = query_page(session, source.url, source.where, offset, size, settings.http_timeout, settings.http_max_retries)
            page_path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(settings.request_delay_s)
        page_features = data.get("features", [])
        features.extend(page_features)
        offset += len(page_features)
        page += 1
        if len(page_features) < size or not page_features:
            break
    return features


def save_normalized(rows: list[NormalizedRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    collection = {"type": "FeatureCollection", "features": [r.to_geojson() for r in rows]}
    path.write_text(json.dumps(collection), encoding="utf-8")


def download_all(settings: Settings, sources: list[Source], force: bool = False) -> dict[str, int]:
    ready = usable_sources(sources)
    skipped = [s.name for s in sources if s.enabled and not s.confirmed]
    if skipped:
        logger.warning("Skipping %d unconfirmed source(s), set confirmed: true after review: %s", len(skipped), skipped)

    session = build_session(settings.http_max_retries)
    counts: dict[str, int] = {}
    for source in progress(ready, "downloading", unit="source", leave=True):
        try:
            features = fetch_pages(session, source, settings, force)
        except (requests.RequestException, ArcGISError) as exc:
            logger.error("%s: download failed - %s", source.name, exc)
            counts[source.name] = 0
            continue

        rows, dropped = normalize_features(source, features)
        save_normalized(rows, NORMALIZED_DIR / f"{source.name}.geojson")
        counts[source.name] = len(rows)
        logger.info("%s: %d fetched, %d kept, %d dropped", source.name, len(features), len(rows), sum(dropped.values()))
    return counts
