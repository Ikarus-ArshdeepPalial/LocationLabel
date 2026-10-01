import numpy as np
import pytest

from label_objs.rampnet import PANO_INPUT_HW, canvas_point_to_image, place_in_panorama

FX, CX, CY = 400.0, 320.0, 240.0


@pytest.mark.parametrize("pitch", [0.0, -12.0, -30.0])
@pytest.mark.parametrize("marker", [(320, 240), (150, 400), (560, 90)])
def test_a_marker_placed_in_the_canvas_maps_back_to_its_pixel(pitch, marker):
    image = np.zeros((480, 640, 3), np.uint8)
    u0, v0 = marker
    image[v0 - 3:v0 + 4, u0 - 3:u0 + 4] = 255

    canvas = place_in_panorama(image, FX, CX, CY, pitch)

    ys, xs = np.nonzero(canvas[..., 0] > 128)
    assert canvas.shape[:2] == PANO_INPUT_HW and len(xs) > 0
    u, v = canvas_point_to_image(xs.mean(), ys.mean(), FX, CX, CY, pitch, image.shape)
    assert (u, v) == pytest.approx((u0, v0), abs=2.0)


def test_canvas_points_outside_the_photo_map_to_none():
    assert canvas_point_to_image(10, 10, FX, CX, CY, 0.0, (480, 640)) is None  # far up-left, behind the camera
    assert canvas_point_to_image(2048, 1024, FX, CX, CY, 0.0, (480, 640)) == pytest.approx((CX, CY), abs=1.0)
