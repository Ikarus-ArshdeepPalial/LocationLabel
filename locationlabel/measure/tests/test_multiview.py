import numpy as np

from measure.config import Settings
from measure.multiview import Reconstruction, estimate_slopes, pick_ramp

SIZE = 60


def grid(x0: float, x1: float, y0: float, y1: float, height) -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(x0, x1, SIZE), np.linspace(y0, y1, SIZE))
    return np.stack([xs, ys, height(xs, ys)], axis=-1)


def reconstruction(slope: float) -> tuple[Reconstruction, list[np.ndarray], list[np.ndarray]]:
    """A flat road (y < 0) and a ramp rising away from the curb with the given slope, seen from two cameras."""
    road = grid(-3, 3, -6, -0.2, lambda x, y: 0 * x)
    ramp = grid(-0.7, 0.7, 0.0, 1.5, lambda x, y: slope * y)
    points = np.zeros((2, SIZE * 2, SIZE, 3))
    points[:, :SIZE] = road
    points[:, SIZE:] = ramp
    ramp_mask = np.zeros((SIZE * 2, SIZE), bool)
    ramp_mask[SIZE:] = True
    centers = np.array([[-3.0, -8.0, 2.5], [3.0, -8.0, 2.5]])
    return Reconstruction(points, np.ones((2, SIZE * 2, SIZE)), centers), [ramp_mask] * 2, [~ramp_mask] * 2


def test_recovers_running_slope_relative_to_the_road_and_ignores_cross_slope():
    rec, ramps, roads = reconstruction(0.083)
    result = estimate_slopes(rec, ramps, roads, [0, 1], 2.5, Settings(output_dir=None))
    assert abs(result.running_pct - 8.3) < 0.5
    assert result.cross_pct < 0.5


def test_a_flat_ramp_measures_flat():
    rec, ramps, roads = reconstruction(0.0)
    result = estimate_slopes(rec, ramps, roads, [0, 1], 2.5, Settings(output_dir=None))
    assert result.running_pct < 0.5


def test_too_few_points_gives_no_estimate():
    rec, ramps, roads = reconstruction(0.05)
    assert estimate_slopes(rec, [np.zeros_like(r) for r in ramps], roads, [0, 1], 2.5, Settings(output_dir=None)) is None


def test_picks_a_segmented_region_over_a_driveway_over_a_bare_point():
    instances = [
        {"class": "curb_ramp", "mask_source": "point", "area_px": 999},
        {"class": "curb_ramp", "mask_source": "driveway", "area_px": 500},
        {"class": "curb_ramp", "mask_source": "segmentation", "area_px": 50},
        {"class": "sidewalk", "area_px": 9999},
    ]
    assert pick_ramp(instances)["mask_source"] == "segmentation"
    assert pick_ramp(instances[:2])["mask_source"] == "driveway"
    assert pick_ramp(instances[:1]) is None


def test_reports_the_width_along_the_curb_and_the_length_up_from_the_road():
    rec, ramps, roads = reconstruction(0.083)
    result = estimate_slopes(rec, ramps, roads, [0, 1], 2.5, Settings(output_dir=None))
    assert abs(result.width_m - 1.4) < 0.15  # 1.4 m wide in the synthetic scene
    assert abs(result.length_m - 1.5) < 0.15


def test_a_confidence_weighted_plane_ignores_doubtful_points():
    import numpy as np
    from measure.planes import fit_plane

    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, (400, 2))
    good = np.c_[xy, 0.05 * xy[:, 0]]                      # a 5 % slope
    bad = np.c_[xy[:150], 0.15 * xy[:150, 0] + 0.002]      # a nearby wrong surface the threshold can't separate
    points = np.r_[good, bad]
    weights = np.r_[np.full(400, 1.0), np.full(150, 0.05)]
    plain = fit_plane(points, 0.2, 50)
    weighted = fit_plane(points, 0.2, 50, weights=weights)
    assert abs(weighted.slope_along(np.array([1.0, 0.0])) - 0.05) < abs(plain.slope_along(np.array([1.0, 0.0])) - 0.05)
    assert abs(weighted.slope_along(np.array([1.0, 0.0])) - 0.05) < 0.005


def test_the_ramp_surface_confidence_drops_when_the_ramp_is_the_doubtful_part():
    import numpy as np
    from measure.multiview import Reconstruction, ramp_surface_confidence

    conf = np.full((2, 10, 10), 4.0)
    ramp = np.zeros((10, 10), bool)
    ramp[3:6, 3:6] = True
    rec = Reconstruction(np.zeros((2, 10, 10, 3)), conf, np.zeros((2, 3)))
    assert ramp_surface_confidence(rec, [ramp, ramp]) == 1.0
    conf[:, 3:6, 3:6] = 2.0
    assert ramp_surface_confidence(rec, [ramp, ramp]) == 0.5
    conf[:, 3:6, 3:6] = 0.1
    assert ramp_surface_confidence(rec, [ramp, ramp]) == 0.3
