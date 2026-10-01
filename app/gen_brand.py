"""Development helper: prepares the Ikarus Delta logo for the app (transparent) and the installer wizard (BMP)."""
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(__file__).parent
src = Image.open(root.parent / "logo" / "image.png").convert("RGB")
rgb = np.asarray(src).astype(float)
alpha = 255 - rgb.min(axis=2)
safe = np.maximum(alpha, 1)[..., None] / 255
color = np.clip((rgb - 255 * (1 - safe)) / safe, 0, 255)
rgba = np.dstack([color, alpha]).astype(np.uint8)
logo = Image.fromarray(rgba, "RGBA")
logo.save(root / "assets" / "ikarus_delta.png")

def on_white(size: tuple[int, int], logo_size: int, name: str) -> None:
    canvas = Image.new("RGB", size, "white")
    scaled = src.resize((logo_size, logo_size), Image.LANCZOS)
    canvas.paste(scaled, ((size[0] - logo_size) // 2, (size[1] - logo_size) // 2))
    canvas.save(root / "assets" / name)

on_white((164, 314), 150, "wizard_large.bmp")
on_white((55, 55), 55, "wizard_small.bmp")
