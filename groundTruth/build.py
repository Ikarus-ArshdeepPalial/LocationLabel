import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from common.settings import ROOT_DIR
from cood_to_sview.geo import distance as geo_distance
from cood_to_sview.provider import GoogleStreetViewProvider, build_session

from .config import CANONICAL_ATTRIBUTES, GROUND_TRUTH_CSV, NORMALIZED_DIR, OBJECT_TYPES, Settings
from .normalize import NormalizedRow
from .panorama import find_nearest_pano, sign_faces_camera
from .sampling import filter_negatives, negative_candidates, stratified_sample, train_test_split
from .sources import Source, usable_sources

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "gt_id", "object_type", "source", "lat", "lng", "is_negative",
    *dict.fromkeys(attr for attrs in CANONICAL_ATTRIBUTES.values() for attr in attrs if attr != "survey_date"),
    "survey_date", "pano_id", "pano_date", "pano_distance_m", "date_gap_years", "split",
]


@dataclass
class GroundTruthRow:
    gt_id: str
    object_type: str
    source: str
    lat: float
    lng: float
    is_negative: bool
    attributes: dict[str, Any] = field(default_factory=dict)
    survey_date: datetime | None = None
    pano_id: str | None = None
    pano_date: str | None = None
    pano_distance_m: float | None = None
    date_gap_years: float | None = None
    split: str = "train"

    def to_csv_row(self) -> dict[str, Any]:
        row = {c: "" for c in CSV_COLUMNS}
        row.update(
            gt_id=self.gt_id, object_type=self.object_type, source=self.source,
            lat=round(self.lat, 7), lng=round(self.lng, 7), is_negative=self.is_negative,
            survey_date=self.survey_date.date().isoformat() if self.survey_date else "",
            pano_id=self.pano_id or "", pano_date=self.pano_date or "",
            pano_distance_m=round(self.pano_distance_m, 2) if self.pano_distance_m is not None else "",
            date_gap_years=round(self.date_gap_years, 2) if self.date_gap_years is not None else "",
            split=self.split,
        )
        for attr, value in self.attributes.items():
            if attr in row:
                row[attr] = value
        return row


def load_normalized(source: Source) -> list[NormalizedRow]:
    path = NORMALIZED_DIR / f"{source.name}.geojson"
    if not path.exists():
        logger.warning("%s: no normalized data, run `groundtruth download` first", source.name)
        return []
    collection = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for feature in collection["features"]:
        props = dict(feature["properties"])
        lat, lng = props.pop("lat"), props.pop("lng")
        props.pop("source", None)
        props.pop("object_type", None)
        rows.append(NormalizedRow(source.name, source.object_type, lat, lng, props, feature.get("geometry")))
    return rows


def parse_pano_date(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        year, month = (int(p) for p in text.split("-", 1))
        return datetime(year, month, 1, tzinfo=timezone.utc)
    except ValueError:
        return None


def date_gap_years(survey: datetime | None, pano: datetime | None) -> float | None:
    if survey is None or pano is None:
        return None
    survey = survey.replace(tzinfo=survey.tzinfo or timezone.utc)
    return abs((pano - survey).days) / 365.25


def locate(provider: GoogleStreetViewProvider, row: NormalizedRow, settings: Settings) -> dict[str, Any] | None:
    found = find_nearest_pano(provider, row.lat, row.lng, settings)
    if found is None:
        return None
    pano, dist = found
    return {
        "pano_id": pano.pano_id, "pano_lat": pano.lat, "pano_lng": pano.lng,
        "pano_date": pano.capture_date, "pano_distance_m": dist, "pano_capture": parse_pano_date(pano.capture_date),
    }


def build_positive_rows(provider: GoogleStreetViewProvider, rows: list[NormalizedRow], object_type: str, settings: Settings) -> list[GroundTruthRow]:
    results = []
    for i, row in enumerate(rows):
        located = locate(provider, row, settings)
        if located is None:
            logger.info("%s #%d: no panorama within %.0f m, skipped", row.source, i, settings.pano_search_radius_m)
            continue

        if object_type in ("stop_sign", "sign_other"):
            facing = row.attributes.get("facing_direction")
            if not sign_faces_camera(row.lat, row.lng, located["pano_lat"], located["pano_lng"], facing, settings.sign_facing_tolerance_deg):
                logger.info("%s #%d: nearest panorama only sees the back of the sign, skipped", row.source, i)
                continue

        gt_id = f"{row.source}_{i:04d}"
        survey = row.attributes.get("survey_date")
        results.append(GroundTruthRow(
            gt_id, object_type, row.source, row.lat, row.lng, is_negative=False,
            attributes={k: v for k, v in row.attributes.items() if k != "survey_date"},
            survey_date=survey, pano_id=located["pano_id"], pano_date=located["pano_date"],
            pano_distance_m=located["pano_distance_m"], date_gap_years=date_gap_years(survey, located["pano_capture"]),
        ))
    return results


def build_negative_rows(provider: GoogleStreetViewProvider, positives: list[GroundTruthRow], all_points: list[tuple[float, float]], object_type: str, settings: Settings, seed: int) -> list[GroundTruthRow]:
    if not positives:
        return []
    count = max(1, round(len(positives) * settings.negative_per_type_fraction))
    anchors = [(p.gt_id, p.lat, p.lng) for p in positives]
    candidates = negative_candidates(anchors, count * 3, settings, seed)  # oversample; some get excluded or lack a pano
    kept = filter_negatives(candidates, all_points, settings.negative_exclusion_radius_m)[:count]

    results = []
    for i, point in enumerate(kept):
        pano = provider.find_pano(point.lat, point.lng, settings.pano_search_radius_m)
        if pano is None:
            continue
        results.append(GroundTruthRow(
            f"neg_{object_type}_{i:04d}", object_type, f"near:{point.near_gt_id}", point.lat, point.lng, is_negative=True,
            pano_id=pano.pano_id, pano_date=pano.capture_date, pano_distance_m=geo_distance(pano.lat, pano.lng, point.lat, point.lng),
        ))
    return results


def run_build(settings: Settings) -> Path:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set; it's needed to look up panoramas (metadata calls are free)")

    session = build_session(settings.http_max_retries)
    provider = GoogleStreetViewProvider(api_key, session, settings.http_timeout)
    sources = usable_sources_by_type()

    all_rows: list[GroundTruthRow] = []
    for object_type in OBJECT_TYPES:
        type_sources = sources.get(object_type, [])
        if not type_sources:
            logger.warning("No confirmed sources for %s, skipping", object_type)
            continue

        rows_by_source = {s.name: load_normalized(s) for s in type_sources}
        positive_sample = stratified_sample(rows_by_source, settings.sample_per_type, settings.random_seed)
        positives = build_positive_rows(provider, positive_sample, object_type, settings)

        all_points = [(r.lat, r.lng) for rows in rows_by_source.values() for r in rows]
        negatives = build_negative_rows(provider, positives, all_points, object_type, settings, settings.random_seed)

        logger.info("%s: %d positive, %d negative", object_type, len(positives), len(negatives))
        all_rows += positives + negatives

    splits = train_test_split([r.gt_id for r in all_rows], settings.train_fraction, settings.random_seed)
    for row in all_rows:
        row.split = splits[row.gt_id]

    GROUND_TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
    with GROUND_TRUTH_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(row.to_csv_row() for row in all_rows)
    logger.info("Wrote %d rows to %s", len(all_rows), GROUND_TRUTH_CSV)
    return GROUND_TRUTH_CSV


def usable_sources_by_type() -> dict[str, list[Source]]:
    from .sources import load_sources

    sources = load_sources()
    return {ot: usable_sources(sources, ot) for ot in OBJECT_TYPES}
