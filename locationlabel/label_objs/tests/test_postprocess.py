import numpy as np

from label_objs.config import CLASS_MAP, CLASSES
from label_objs.models import Box
from label_objs.postprocess import (
    associate_pole,
    canonical_map,
    curb_line,
    lowest_pixel,
    map_labels,
    split_by_confidence,
    split_instances,
)


def test_map_labels_matches_case_insensitively_and_reports_unmatched():
    id2label = {0: "road", 1: "Sidewalk", 7: "Curb Cut", 9: "Traffic Sign Frame", 12: "Sky"}

    id_to_class, unmatched = map_labels(id2label, CLASS_MAP)

    assert id_to_class == {0: "road", 1: "sidewalk", 7: "curb_ramp", 9: "pole"}
    assert "curb" in unmatched and "fire_hydrant" in unmatched
    assert "curb_ramp" not in unmatched


def test_canonical_map_uses_class_indices():
    labels = np.array([[0, 7], [12, 1]])

    mapped = canonical_map(labels, {0: "road", 1: "sidewalk", 7: "curb_ramp"}, CLASSES)

    assert mapped.tolist() == [[CLASSES.index("road"), CLASSES.index("curb_ramp")], [0, CLASSES.index("sidewalk")]]


def test_split_instances_separates_components_and_drops_small_ones():
    mask = np.zeros((100, 100), bool)
    mask[10:30, 10:30] = True   # ramp 1: 400 px
    mask[10:30, 60:80] = True   # ramp 2: 400 px
    mask[80:83, 80:83] = True   # speck: 9 px

    instances = split_instances(mask, min_area=50)

    assert len(instances) == 2
    assert sorted(int(i.sum()) for i in instances) == [400, 400]


def test_split_instances_closes_one_pixel_gaps():
    mask = np.zeros((50, 50), bool)
    mask[10:40, 10:40] = True
    mask[25, 10:40] = False

    assert len(split_instances(mask, min_area=10)) == 1


def pole_mask(x1: int, x2: int, y1: int, y2: int) -> np.ndarray:
    mask = np.zeros((200, 200), bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_associate_pole_picks_pole_beneath_sign():
    sign = (30.0, 8.0, 54.0, 32.0)
    poles = [pole_mask(150, 154, 20, 150), pole_mask(40, 44, 30, 140)]

    assert associate_pole(sign, poles) == 1
    assert lowest_pixel(poles[1]) == [42, 139]


def test_associate_pole_ignores_poles_beside_or_far_below():
    sign = (30.0, 8.0, 54.0, 32.0)
    beside = pole_mask(120, 124, 0, 150)
    far_below = pole_mask(40, 44, 120, 190)

    assert associate_pole(sign, [beside, far_below]) is None


def test_curb_line_follows_road_boundary():
    sidewalk = np.zeros((200, 200), bool)
    sidewalk[:100, 20:180] = True
    road = np.zeros((200, 200), bool)
    road[100:] = True

    line = curb_line(sidewalk, road)

    xs = [x for x, _ in line]
    assert xs == sorted(xs) and xs[0] <= 20 and xs[-1] >= 179
    assert all(y in (100, 101) for _, y in line)
    assert len(line) <= 65


def test_curb_line_empty_when_not_touching_road():
    instance = np.zeros((50, 50), bool)
    instance[:10] = True
    road = np.zeros((50, 50), bool)
    road[40:] = True

    assert curb_line(instance, road) == []


def test_split_by_confidence():
    boxes = [Box("stop_sign", 0.9, (0, 0, 1, 1)), Box("stop_sign", 0.39, (0, 0, 1, 1)), Box("stop_sign", 0.4, (0, 0, 1, 1))]

    kept, dropped = split_by_confidence(boxes, 0.4)

    assert [b.score for b in kept] == [0.9, 0.4]
    assert [b.score for b in dropped] == [0.39]
