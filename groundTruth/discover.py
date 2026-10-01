"""The `groundtruth inspect` command: check reachability, print fields/samples, propose mappings."""

import logging

import requests

from common.http import build_session

from .arcgis import LayerInfo, layer_info, sample_features
from .config import CANONICAL_ATTRIBUTES, Settings
from .sources import FieldMapping, Source, load_sources, save_sources

logger = logging.getLogger(__name__)

# attribute -> keywords tried in order against lowercased, underscore-stripped field names
KEYWORD_HINTS: dict[str, list[str]] = {
    "running_slope_pct": ["runningslope", "slope"],
    "cross_slope_pct": ["crossslope", "xslope"],
    "ramp_width_in": ["rampwidth", "openingwidth", "width"],
    "gutter_lip_height_in": ["gutterlip", "gutterheight", "lipheight"],
    "width_in": ["width"],
    "condition": ["condition", "pavementindex", "score"],
    "sign_code": ["signtype", "signcode", "mutcd", "type"],
    "sign_size_in": ["signsz", "signsize", "size"],
    "facing_direction": ["facing", "direction"],
    "side_of_street": ["side"],
    "survey_date": ["surveydate", "inspection", "inspectdate", "installdate", "instdate", "date"],
}


def propose_mapping(object_type: str, fields: list[dict[str, str]]) -> dict[str, FieldMapping]:
    names = [f["name"] for f in fields]
    normalized = {n: n.lower().replace("_", "") for n in names}
    proposal: dict[str, FieldMapping] = {}
    for attr in CANONICAL_ATTRIBUTES[object_type]:
        for keyword in KEYWORD_HINTS.get(attr, []):
            match = min((n for n, norm in normalized.items() if keyword in norm), key=len, default=None)
            if match:
                proposal[attr] = FieldMapping(source_field=match)
                break
    return proposal


def inspect_source(session: requests.Session, source: Source, sample_size: int, settings: Settings) -> LayerInfo:
    info = layer_info(session, source.url, settings.http_timeout)
    print(f"\n=== {source.name} ({source.city}, {source.state}) — {source.object_type}")
    print(f"    {source.url}")
    if not info.reachable:
        print(f"    UNREACHABLE: {info.error}")
        return info

    print(f"    {len(info.fields)} fields, geometry={info.geometry_type}, page size={info.max_record_count}")
    for f in info.fields:
        print(f"      {f['name']}: {f['type']}")

    try:
        samples = sample_features(session, source.url, source.where, sample_size, settings.http_timeout)
    except (requests.RequestException, Exception) as exc:  # noqa: BLE001 - report and continue
        print(f"    sample query failed: {exc}")
        samples = []
    for feature in samples:
        print(f"    sample: {feature.get('geometry')}")
        print(f"            {feature.get('attributes')}")

    if not source.field_mapping:
        proposed = propose_mapping(source.object_type, info.fields)
        if proposed:
            print(f"    proposed field_mapping (confirmed: false): {[a for a in proposed]}")
            source.field_mapping = proposed
    return info


def cmd_inspect(settings: Settings, only: list[str] | None = None, sample_size: int = 3) -> None:
    sources = load_sources()
    session = build_session(settings.http_max_retries)
    changed = False
    for source in sources:
        if only and source.name not in only:
            continue
        if not source.enabled:
            print(f"\n=== {source.name}: disabled, skipped")
            continue
        before = dict(source.field_mapping)
        inspect_source(session, source, sample_size, settings)
        if source.field_mapping != before:
            changed = True
        status = "confirmed" if source.confirmed else "NOT CONFIRMED — review field_mapping, then set confirmed: true"
        print(f"    {status}")
    if changed:
        save_sources(sources)
        logger.info("Wrote proposed field mappings to groundTruth/config/groundtruth_sources.yaml")
