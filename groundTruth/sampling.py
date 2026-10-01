import random
from dataclasses import dataclass

from cood_to_sview.geo import destination, distance

from .config import Settings
from .normalize import NormalizedRow


def stratified_sample(rows_by_source: dict[str, list[NormalizedRow]], n: int, seed: int) -> list[NormalizedRow]:
    """Up to n rows, round-robin across sources so no single source dominates; deterministic for a given seed."""
    rng = random.Random(seed)
    remaining = {name: rng.sample(rows, len(rows)) for name, rows in sorted(rows_by_source.items()) if rows}
    names = list(remaining)
    picked: list[NormalizedRow] = []
    i = 0
    while len(picked) < n and remaining:
        name = names[i % len(names)]
        if remaining[name]:
            picked.append(remaining[name].pop(0))
            if not remaining[name]:
                del remaining[name]
                names = list(remaining)
                i = -1  # restart the round so removal doesn't skip a source
        i += 1
    return picked


@dataclass
class NegativePoint:
    lat: float
    lng: float
    near_gt_id: str


def negative_candidates(anchors: list[tuple[str, float, float]], count: int, settings: Settings, seed: int) -> list[NegativePoint]:
    """Points `negative_min/max_distance_m` from a random anchor, in a random direction."""
    rng = random.Random(seed)
    picks = []
    for _ in range(count):
        gt_id, lat, lng = anchors[rng.randrange(len(anchors))]
        bearing = rng.uniform(0, 360)
        dist = rng.uniform(settings.negative_min_distance_m, settings.negative_max_distance_m)
        new_lat, new_lng = destination(lat, lng, bearing, dist)
        picks.append(NegativePoint(new_lat, new_lng, gt_id))
    return picks


def far_from_all(point: tuple[float, float], exclusion_points: list[tuple[float, float]], radius_m: float) -> bool:
    return all(distance(point[0], point[1], p[0], p[1]) >= radius_m for p in exclusion_points)


def filter_negatives(candidates: list[NegativePoint], exclusion_points: list[tuple[float, float]], radius_m: float) -> list[NegativePoint]:
    return [c for c in candidates if far_from_all((c.lat, c.lng), exclusion_points, radius_m)]


def train_test_split(gt_ids: list[str], train_fraction: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = sorted(gt_ids)
    rng.shuffle(shuffled)
    cut = round(len(shuffled) * train_fraction)
    train, test = set(shuffled[:cut]), set(shuffled[cut:])
    return {gt_id: ("train" if gt_id in train else "test") for gt_id in gt_ids}
