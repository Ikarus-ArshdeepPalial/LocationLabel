import json
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image

from user_images.calibration import Calibration, Calibrator


def textured(width: int, height: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)


def save_jpeg(path: Path, pixels: np.ndarray, exif: Image.Exif | None = None) -> Path:
    Image.fromarray(pixels).save(path, "JPEG", exif=exif or Image.Exif())
    return path


def phone_exif(focal_35mm: int = 26, orientation: int = 1, lat=(30, 9, 16.8), lng=(97, 51, 18.0), heading: float | None = 123.5) -> Image.Exif:
    """EXIF like a phone writes: 35 mm equivalent focal length, GPS position/direction, orientation."""
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.FocalLengthIn35mmFilm] = focal_35mm
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.DateTimeOriginal] = "2026:05:01 10:00:00"
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps[ExifTags.GPS.GPSLatitudeRef], gps[ExifTags.GPS.GPSLatitude] = "N", lat
    gps[ExifTags.GPS.GPSLongitudeRef], gps[ExifTags.GPS.GPSLongitude] = "W", lng
    if heading is not None:
        gps[ExifTags.GPS.GPSImgDirectionRef], gps[ExifTags.GPS.GPSImgDirection] = "T", heading
    return exif


def write_json(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values))


class FakeCalibrator(Calibrator):
    def __init__(self, name: str, result: Calibration | None) -> None:
        self.name = name
        self.result = result
        self.calls = 0

    def estimate(self, image: np.ndarray, focal_px: float | None) -> Calibration | None:
        self.calls += 1
        return self.result


def vertical_lines_image(pitch_deg: float, roll_deg: float = 0.0, size: int = 640, focal: float = 400.0) -> np.ndarray:
    """Render vertical world poles seen by a camera with the given pitch, then roll the image."""
    from measure.geometry import Camera
    from measure.tests.synthetic import project

    camera = Camera(focal, focal, size / 2, size / 2, heading=0, pitch=pitch_deg, height=1.5)
    image = np.full((size, size, 3), 255, np.uint8)
    for x in np.linspace(-6, 6, 9):
        for depth in (8.0, 14.0):
            top = project(camera, [x, depth, 4.0])
            bottom = project(camera, [x, depth, 0.0])
            cv2.line(image, tuple(int(round(v)) for v in top), tuple(int(round(v)) for v in bottom), (0, 0, 0), 3)
    if roll_deg:
        # Tilt the image's verticals clockwise by roll_deg, as a camera rolled the other way would.
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), -roll_deg, 1.0)
        image = cv2.warpAffine(image, matrix, (size, size), borderValue=(255, 255, 255))
    return image
