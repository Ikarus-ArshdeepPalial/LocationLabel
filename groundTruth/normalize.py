import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .fields import is_plausible, map_attributes
from .geometry import representative_point
from .sources import Source

logger = logging.getLogger(__name__)


@dataclass
class NormalizedRow:
    source: str
    object_type: str
    lat: float
    lng: float
    attributes: dict[str, Any]
    geometry: dict[str, Any] | None

    def to_geojson(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": self.geometry or {"type": "Point", "coordinates": [self.lng, self.lat]},
            "properties": {"source": self.source, "object_type": self.object_type, "lat": self.lat, "lng": self.lng, **self.attributes},
        }


def normalize_features(source: Source, features: list[dict]) -> tuple[list[NormalizedRow], Counter]:
    """Canonical rows plus a Counter of drop reasons (no geometry / implausible attribute)."""
    rows, dropped = [], Counter()
    for feature in features:
        point = representative_point(feature.get("geometry"))
        if point is None:
            dropped["no geometry"] += 1
            continue

        canonical = map_attributes(feature.get("attributes", {}), source.field_mapping)
        ok, category, detail = is_plausible(source.object_type, canonical)
        if not ok:
            dropped[category] += 1
            if dropped[category] == 1:
                logger.debug("%s: example drop (%s): %s", source.name, category, detail)
            continue

        rows.append(NormalizedRow(source.name, source.object_type, point[0], point[1], canonical, feature.get("geometry")))

    for category, count in dropped.items():
        logger.info("%s: dropped %d record(s) for implausible %s", source.name, count, category)
    return rows, dropped
