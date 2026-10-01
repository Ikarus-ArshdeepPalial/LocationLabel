from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw

THUMB = 320
LABEL_HEIGHT = 22
COLUMNS = 4


def make_contact_sheet(images: list[tuple[Path, str]], out_path: Path) -> None:
    columns = min(COLUMNS, len(images))
    rows = ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * THUMB, rows * (THUMB + LABEL_HEIGHT)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (path, label) in enumerate(images):
        x, y = (i % columns) * THUMB, (i // columns) * (THUMB + LABEL_HEIGHT)
        with Image.open(path) as img:
            sheet.paste(img.convert("RGB").resize((THUMB, THUMB)), (x, y + LABEL_HEIGHT))
        draw.text((x + 4, y + 5), label, fill="black")

    sheet.save(out_path, quality=90)
