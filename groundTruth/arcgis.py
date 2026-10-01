"""Minimal client for ArcGIS FeatureServer/MapServer layers: metadata, sampling, and paginated queries."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ArcGISError(Exception):
    pass


@dataclass
class LayerInfo:
    name: str
    geometry_type: str
    max_record_count: int
    fields: list[dict[str, str]]
    reachable: bool
    error: str | None = None


def _get_json(session: requests.Session, url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = session.get(url, params={**params, "f": "json"}, timeout=timeout)
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as exc:
        raise ArcGISError(f"non-JSON response from {url} (likely blocked or wrong URL)") from exc
    if isinstance(data, dict) and "error" in data:
        raise ArcGISError(str(data["error"]))
    return data


def layer_info(session: requests.Session, layer_url: str, timeout: float = 20) -> LayerInfo:
    try:
        data = _get_json(session, layer_url, {}, timeout)
    except (requests.RequestException, ArcGISError) as exc:
        return LayerInfo("", "", 0, [], reachable=False, error=str(exc))

    fields = [{"name": f["name"], "type": f["type"].replace("esriFieldType", "")} for f in data.get("fields", [])]
    return LayerInfo(
        name=data.get("name", ""),
        geometry_type=data.get("geometryType", ""),
        max_record_count=int(data.get("maxRecordCount") or 1000),
        fields=fields,
        reachable=True,
    )


def sample_features(session: requests.Session, layer_url: str, where: str = "1=1", n: int = 3, timeout: float = 20) -> list[dict]:
    data = _get_json(session, f"{layer_url}/query", {"where": where, "outFields": "*", "outSR": 4326, "resultRecordCount": n}, timeout)
    return data.get("features", [])


def query_page(
    session: requests.Session, layer_url: str, where: str, offset: int, size: int, timeout: float, max_retries: int
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return _get_json(
                session, f"{layer_url}/query",
                {"where": where, "outFields": "*", "outSR": 4326, "resultOffset": offset, "resultRecordCount": size},
                timeout,
            )
        except (requests.RequestException, ArcGISError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                logger.warning("Query failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
                time.sleep(2**attempt)
    raise last_exc  # type: ignore[misc]


def query_all(
    session: requests.Session,
    layer_url: str,
    where: str = "1=1",
    page_size: int | None = None,
    timeout: float = 30,
    delay: float = 0.2,
    max_retries: int = 3,
    on_page: "callable[[int, dict], None] | None" = None,
) -> list[dict]:
    """All features, paginating with resultOffset; on_page(page_index, raw_page) fires for each request."""
    info = layer_info(session, layer_url, timeout)
    if not info.reachable:
        raise ArcGISError(info.error or "layer unreachable")
    size = min(page_size or info.max_record_count, info.max_record_count) or 1000

    features: list[dict] = []
    offset, page = 0, 0
    while True:
        data = query_page(session, layer_url, where, offset, size, timeout, max_retries)
        if on_page:
            on_page(page, data)
        page_features = data.get("features", [])
        features.extend(page_features)
        offset += len(page_features)
        page += 1
        if len(page_features) < size or not page_features:
            break
        time.sleep(delay)
    return features
