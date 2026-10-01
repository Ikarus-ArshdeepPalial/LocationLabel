from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_real_depth_model_on_sample_street_image():
    import ultralytics

    from label_objs.models import select_device
    from measure.config import Settings
    from measure.depth import HFDepthModel

    image = cv2.cvtColor(cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg")), cv2.COLOR_BGR2RGB)

    depth = HFDepthModel(Settings.depth_model, select_device()).predict(image)

    assert depth.shape == image.shape[:2]
    assert np.isfinite(depth).all()
    assert 0.5 < np.median(depth) < 80
    assert depth[-10:].mean() < depth[:10].mean()  # street at the bottom is nearer than the sky/buildings at the top
