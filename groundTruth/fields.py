"""Turn a raw source record into canonical attributes: string parsing, unit conversion, plausibility checks."""

import logging
import re
from datetime import datetime, timezone
from typing import Any

from .sources import FieldMapping

logger = logging.getLogger(__name__)

PERCENT_UNITS = {"percent": 1.0, "ratio": 100.0}
INCH_UNITS = {"inch": 1.0, "foot": 12.0, "cm": 1 / 2.54, "mm": 1 / 25.4, "meter": 1 / 0.0254}

MUTCD_CODE = re.compile(r"^\s*([A-Z]{1,2}\d{1,3}(?:-\d+[A-Za-z]{0,2})?)")
SIGN_SIZE = re.compile(r"(\d+(?:\.\d+)?)")
COMPASS_DEGREES = {
    "N": 0, "NORTH": 0, "NNE": 22.5, "NE": 45, "NORTHEAST": 45, "ENE": 67.5,
    "E": 90, "EAST": 90, "ESE": 112.5, "SE": 135, "SOUTHEAST": 135, "SSE": 157.5,
    "S": 180, "SOUTH": 180, "SSW": 202.5, "SW": 225, "SOUTHWEST": 225, "WSW": 247.5,
    "W": 270, "WEST": 270, "WNW": 292.5, "NW": 315, "NORTHWEST": 315, "NNW": 337.5,
}
MUTCD_STOP_SIGN_SIZES_IN = {18, 24, 30, 36, 48}

PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "running_slope_pct": (0, 30),
    "cross_slope_pct": (0, 30),
    "ramp_width_in": (12, 96),
    "gutter_lip_height_in": (0, 4),
    "width_in": (12, 240),
    "facing_direction": (0, 360),
}


def to_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in ("N/A", "NA", "NULL", "NONE"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def convert_unit(value: float, unit: str | None) -> float | None:
    if unit is None:
        return value
    if unit in PERCENT_UNITS:
        return value * PERCENT_UNITS[unit]
    if unit in INCH_UNITS:
        return value * INCH_UNITS[unit]
    raise ValueError(f"unknown unit {unit!r}")


def parse_mutcd_code(value: Any) -> str | None:
    match = MUTCD_CODE.match(str(value)) if value else None
    return match.group(1) if match else None


def parse_sign_size_inches(value: Any) -> float | None:
    if not value:
        return None
    numbers = [float(n) for n in SIGN_SIZE.findall(str(value))]
    return numbers[0] if numbers else None


def parse_compass_to_degrees(value: Any) -> float | None:
    if not value:
        return None
    return COMPASS_DEGREES.get(str(value).strip().upper())


def parse_epoch_ms(value: Any, min_year: int = 1970) -> datetime | None:
    number = to_number(value)
    if number is None:
        return None
    try:
        when = datetime.fromtimestamp(number / 1000, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    return when if when.year >= min_year else None


def parse_date_string(value: Any, fmt: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), fmt)
    except ValueError:
        return None


def apply_parse(name: str, value: Any) -> Any:
    if name == "mutcd_code":
        return parse_mutcd_code(value)
    if name == "sign_size_inches":
        return parse_sign_size_inches(value)
    if name == "compass_to_degrees":
        return parse_compass_to_degrees(value)
    if name.startswith("epoch_ms"):
        min_year = int(name.split("min_year=")[1]) if "min_year=" in name else 1970
        return parse_epoch_ms(value, min_year)
    if name.startswith("date_string:"):
        return parse_date_string(value, name.split(":", 1)[1])
    raise ValueError(f"unknown parser {name!r}")


def extract_attribute(attrs: dict[str, Any], mapping: FieldMapping) -> Any:
    for field_name in [mapping.source_field, *mapping.fallback]:
        if field_name is None:
            continue
        raw = attrs.get(field_name)
        if raw is None or raw == "":
            continue
        if mapping.parse:
            value = apply_parse(mapping.parse, raw)
        elif mapping.unit:
            number = to_number(raw)
            value = convert_unit(number, mapping.unit) if number is not None else None
        else:
            value = raw  # a plain field with no unit/parser is copied through as-is (e.g. a text code)
        if value is not None:
            return value
    return None


def map_attributes(attrs: dict[str, Any], field_mapping: dict[str, FieldMapping]) -> dict[str, Any]:
    return {attr: extract_attribute(attrs, mapping) for attr, mapping in field_mapping.items()}


def is_plausible(object_type: str, canonical: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    """(ok, category, detail) — category groups reasons for counting; both are None when the row passes."""
    for attr, value in canonical.items():
        if value is None or attr == "survey_date":
            continue
        if attr == "sign_size_in" and object_type == "stop_sign":
            if value not in MUTCD_STOP_SIGN_SIZES_IN:
                return False, "sign_size_in", f"sign_size_in={value} not a standard MUTCD size {sorted(MUTCD_STOP_SIGN_SIZES_IN)}"
            continue
        bounds = PLAUSIBLE_RANGES.get(attr)
        if bounds and not (bounds[0] <= value <= bounds[1]):
            return False, attr, f"{attr}={value} outside plausible range {bounds}"
    return True, None, None
