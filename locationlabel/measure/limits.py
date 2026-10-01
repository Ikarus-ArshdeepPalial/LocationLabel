"""Regional limits: a measured value is compared with a limit, and only counts as over it once it passes the margin.

A limit is a rule such as "driveway slope, at most 10 %, margin 0.5 %". Slopes are compared in percent and lengths in
metres internally; each rule is stored in whatever unit its author chose (%, degrees, cm, m, in, ft).
"""

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# measure id -> (label, "max" or "min", "slope" or "length")
MEASURES = {
    "ramp_running_slope": ("Curb ramp: slope along the ramp", "max", "slope"),
    "ramp_cross_slope": ("Curb ramp: cross slope", "max", "slope"),
    "driveway_running_slope": ("Driveway: slope toward the street", "max", "slope"),
    "driveway_min_width": ("Driveway: width, at least", "min", "length"),
    "driveway_max_width": ("Driveway: width, at most", "max", "length"),
    "stop_sign_bottom_height": ("Stop sign: height to the bottom of the sign", "min", "length"),
    "stop_sign_total_height": ("Stop sign: total height (to the top of the sign)", "max", "length"),
    "pole_height": ("Pole: height", "max", "length"),
    "curb_height": ("Curb: height", "max", "length"),
    "sidewalk_width": ("Sidewalk: width", "min", "length"),
}
# The Limits page groups measures by what they are about: section -> [(measure, short name)].
SECTIONS = {
    "Sidewalk": [("sidewalk_width", "Width"), ("curb_height", "Curb height")],
    "Ramp": [("ramp_running_slope", "Slope along the ramp"), ("ramp_cross_slope", "Cross slope")],
    "Driveway": [("driveway_running_slope", "Slope"), ("driveway_min_width", "Width, at least"), ("driveway_max_width", "Width, at most")],
    "Pole": [("pole_height", "Height")],
    "Signs": [("stop_sign_bottom_height", "Height to the bottom of the sign"), ("stop_sign_total_height", "Total height, including the sign")],
}
SECTION_OF = {measure: section for section, items in SECTIONS.items() for measure, _ in items}
UNITS = {"slope": ("%", "°"), "length": ("cm", "m", "in", "ft")}
METRES = {"cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}
DEFAULT_MARGIN = {"%": 0.5, "°": 0.3, "cm": 2.0, "m": 0.02, "in": 1.0, "ft": 0.1}


def to_base(value: float, unit: str) -> float:
    """A slope as percent, or a length as metres."""
    if unit == "%":
        return value
    if unit == "°":
        return math.tan(math.radians(value)) * 100
    return value * METRES[unit]


def from_base(base: float, unit: str) -> float:
    if unit == "%":
        return base
    if unit == "°":
        return math.degrees(math.atan(base / 100))
    return base / METRES[unit]


def format_value(value: float, unit: str) -> str:
    if unit in ("%", "°"):
        return f"{value:.1f} %" if unit == "%" else f"{value:.1f}°"
    if unit == "cm":
        return f"{round(value)} cm"
    return f"{value:.2f} m" if unit == "m" else f"{value:.1f} {unit}" if unit == "ft" else f"{value:.0f} {unit}"


@dataclass
class Limit:
    name: str
    measure: str
    limit: float
    unit: str
    margin: float = 0.0
    source: str = ""
    enabled: bool = True

    @property
    def kind(self) -> str:
        return MEASURES[self.measure][1]

    @property
    def limit_text(self) -> str:
        return format_value(self.limit, self.unit)

    @property
    def limit_base(self) -> float:
        return to_base(self.limit, self.unit)

    @property
    def margin_base(self) -> float:
        """The margin in base units; degrees are not linear, so it is the difference between the two angles."""
        direction = 1 if self.kind == "max" else -1
        return abs(to_base(self.limit + direction * self.margin, self.unit) - self.limit_base)


@dataclass
class Check:
    rule: Limit
    subject: str
    value_base: float
    exceeded: bool
    certain: bool  # False when the value is a low-confidence estimate

    @property
    def value(self) -> float:
        return from_base(self.value_base, self.rule.unit)

    @property
    def is_slope(self) -> bool:
        return MEASURES[self.rule.measure][2] == "slope"

    @property
    def value_text(self) -> str:
        """What is drawn big on the image: slopes in degrees (whatever unit the limit uses), lengths in the limit's unit."""
        if self.is_slope:
            return format_value(from_base(self.value_base, "°"), "°")
        return format_value(self.value, self.rule.unit)

    @property
    def message(self) -> str:
        bound = f"{'max' if self.rule.kind == 'max' else 'min'} {self.rule.limit_text}"
        if self.rule.kind == "max":
            text = f"limit exceeded: {bound}" if self.certain else f"may exceed the limit: {bound}"
        else:
            text = f"below the limit: {bound}" if self.certain else f"may be below the limit: {bound}"
        return f"{format_value(self.value_base, '%')} · {text}" if self.is_slope else text  # the percent goes in the description

    def to_dict(self) -> dict:
        return {
            "rule": self.rule.name,
            "measure": self.rule.measure,
            "subject": self.subject,
            "value": round(self.value, 3),
            "limit": self.rule.limit,
            "unit": self.rule.unit,
            "margin": self.rule.margin,
            "kind": self.rule.kind,
            "exceeded": self.exceeded,
            "certain": self.certain,
            "source": self.rule.source,
        }


def evaluate(limits: list[Limit], measure: str, value_base: float, subject: str, certain: bool = True) -> list[Check]:
    """Checks one value against every enabled rule for that measure. Over the limit by no more than the margin is not exceeded."""
    checks = []
    for rule in limits:
        if not rule.enabled or rule.measure != measure:
            continue
        over = value_base - rule.limit_base if rule.kind == "max" else rule.limit_base - value_base
        checks.append(Check(rule, subject, value_base, over > rule.margin_base, certain))
    return checks


# What a new installation starts with. Everything here can be edited, switched off or deleted in the app.
DEFAULTS = [
    Limit("Driveway slope (Austin)", "driveway_running_slope", 10.0, "%", 0.5, "Austin Standard 433S-1A"),
    Limit("Driveway width, at least (Austin)", "driveway_min_width", 10.0, "ft", 0.5, "Austin Standard 433S-1A"),
    Limit("Driveway width, at most (Austin)", "driveway_max_width", 20.0, "ft", 0.5, "Austin Standard 433S-1A"),
    Limit("Stop sign height (Texas MUTCD)", "stop_sign_bottom_height", 7.0, "ft", 0.2, "Texas MUTCD: 7 ft to the bottom of the sign"),
    Limit("Sidewalk width (ADA / PROWAG)", "sidewalk_width", 4.0, "ft", 0.3, "ADA / PROWAG: 4 ft clear width"),
    Limit("Curb height, at most", "curb_height", 9.0, "in", 0.5, "the pipeline's default maximum: set your own standard"),
]


def load_limits(path: Path | str | None) -> list[Limit]:
    """Limits from a JSON list. A missing or unreadable file means no limits; a bad entry is skipped."""
    if not path:
        return []
    try:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Could not read the limits file %s", path)
        return []
    limits = []
    for entry in entries if isinstance(entries, list) else []:
        try:
            rule = Limit(**{k: entry[k] for k in ("name", "measure", "limit", "unit") if k in entry}, **{k: entry[k] for k in ("margin", "source", "enabled") if k in entry})
            if rule.measure in MEASURES and rule.unit in UNITS[MEASURES[rule.measure][2]]:
                limits.append(rule)
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipped an invalid limit: %r", entry)
    return limits


def save_limits(path: Path, limits: list[Limit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(rule) for rule in limits], indent=2), encoding="utf-8")
