from dataclasses import dataclass
from math import hypot
from typing import Any

import numpy as np
from PIL import ExifTags, Image, ImageOps

# 35 mm "equivalent" focal lengths are defined against the 36x24 mm frame's diagonal.
FULL_FRAME_DIAGONAL_MM = hypot(36, 24)


@dataclass
class ExifInfo:
    focal_px: float | None = None
    lat: float | None = None
    lng: float | None = None
    heading_deg: float | None = None
    capture_date: str | None = None
    orientation: int = 1


def load_oriented(path) -> tuple[np.ndarray, Image.Exif]:
    """RGB pixels rotated/flipped as the EXIF Orientation tag says, plus the raw EXIF."""
    with Image.open(path) as image:
        exif = image.getexif()
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")), exif


def to_degrees(dms: Any, ref: str | None) -> float:
    degrees, minutes, seconds = (float(x) for x in dms)
    value = degrees + minutes / 60 + seconds / 3600
    return -value if ref in ("S", "W") else value


def parse_exif(exif: Image.Exif, width: int, height: int) -> ExifInfo:
    """Camera facts from EXIF; width/height are the oriented image size."""
    details = exif.get_ifd(ExifTags.IFD.Exif)
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    info = ExifInfo(orientation=int(exif.get(ExifTags.Base.Orientation, 1)))

    focal_35mm = details.get(ExifTags.Base.FocalLengthIn35mmFilm)
    if focal_35mm:
        info.focal_px = float(focal_35mm) * hypot(width, height) / FULL_FRAME_DIAGONAL_MM

    if ExifTags.GPS.GPSLatitude in gps and ExifTags.GPS.GPSLongitude in gps:
        info.lat = to_degrees(gps[ExifTags.GPS.GPSLatitude], gps.get(ExifTags.GPS.GPSLatitudeRef))
        info.lng = to_degrees(gps[ExifTags.GPS.GPSLongitude], gps.get(ExifTags.GPS.GPSLongitudeRef))
    if ExifTags.GPS.GPSImgDirection in gps:
        # Magnetic headings ('M') are kept as-is; declination is at most ~20 degrees in the contiguous US.
        info.heading_deg = float(gps[ExifTags.GPS.GPSImgDirection]) % 360

    date = details.get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
    info.capture_date = str(date) if date else None
    return info
