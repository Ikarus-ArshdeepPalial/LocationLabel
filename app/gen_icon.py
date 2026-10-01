"""Development helper: draws assets/app.ico (a purple tile with a white ramp wedge)."""
from pathlib import Path
from PIL import Image, ImageDraw

size = 512
image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=112, fill=(142, 68, 214, 255))
draw.polygon([(96, 380), (416, 380), (416, 236)], fill=(255, 255, 255, 255))
draw.rectangle((96, 380, 416, 412), fill=(255, 255, 255, 200))
draw.ellipse((120, 110, 200, 190), fill=(255, 255, 255, 255))
image.save(Path(__file__).parent / "assets" / "app.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
