"""A synthetic curb ramp seen from the road: a scene with a known slope, rendered from chosen camera positions.

World axes: x along the curb, y away from the road (toward the house), z up. The road is z=0 for y<0, the curb
face is at y=0, the sidewalk is at z=curb height, and the ramp rises from the gutter to the sidewalk at `slope`.
Labels per pixel: 0 sky, 1 road, 2 ramp, 3 curb face and ramp sides, 4 sidewalk, 5 grass.
"""

from dataclasses import dataclass

import cv2
import numpy as np

ROAD, RAMP, CURB, SIDEWALK, GRASS = 1, 2, 3, 4, 5


def value_noise(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    grid = rng.random((cells + 1, cells + 1)).astype(np.float32)
    return cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)


def make_texture(kind: str, seed: int, size: int = 1024) -> np.ndarray:
    """Tileable-looking noise texture, rich enough for feature matching; 100 pixels per metre."""
    rng = np.random.default_rng(seed)
    base = sum(value_noise(size, cells, rng) * weight for cells, weight in ((4, 0.45), (16, 0.3), (64, 0.2), (256, 0.15)))
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    if kind == "grass":
        return np.clip(np.stack([0.20 + 0.15 * base, 0.35 + 0.25 * base, 0.12 + 0.1 * base], axis=-1), 0, 1).astype(np.float32)
    if kind == "asphalt":
        gray = 0.18 + 0.22 * base + 0.08 * (rng.random((size, size)) > 0.985)
        tint = np.array([1.0, 1.0, 1.05])
    else:  # concrete, with expansion joints every 1.5 m
        gray = 0.55 + 0.25 * base
        gray[::150, :] *= 0.55
        gray[:, ::150] *= 0.55
        tint = np.array([1.0, 0.98, 0.94])
    return np.clip(gray[..., None] * tint, 0, 1).astype(np.float32)


@dataclass
class Scene:
    slope: float            # rise over run of the ramp
    curb_height: float = 0.12
    ramp_half_width: float = 0.75
    seed: int = 0
    sidewalk_width: float = 1.5

    def __post_init__(self) -> None:
        self.asphalt = make_texture("asphalt", self.seed)
        self.concrete = make_texture("concrete", self.seed + 1)
        self.grass = make_texture("grass", self.seed + 2)

    @property
    def ramp_length(self) -> float:
        return self.curb_height / self.slope


def look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Columns are the camera's right, down and forward directions in world coordinates."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


def sample(texture: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    size = texture.shape[0]
    return texture[(b * 100).astype(int) % size, (a * 100).astype(int) % size]


def render(scene: Scene, position: np.ndarray, target: np.ndarray, focal: float, width: int, height: int, supersample: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(image float RGB, label map, camera rotation) for a pinhole camera with the principal point at the centre."""
    rotation = look_at(np.asarray(position, float), np.asarray(target, float))
    w, h = width * supersample, height * supersample
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - w / 2) / (focal * supersample), (v - h / 2) / (focal * supersample), np.ones_like(u)], axis=-1)
    d = rays @ rotation.T
    c = np.asarray(position, float)

    s, hw, hc, length = scene.slope, scene.ramp_half_width, scene.curb_height, scene.ramp_length
    best = np.full(d.shape[:2], np.inf)
    label = np.zeros(d.shape[:2], np.uint8)
    color = np.full(d.shape, 0.7, np.float32)
    color[..., 2] = 0.9  # sky

    def consider(t: np.ndarray, valid: np.ndarray, lab: int, rgb: np.ndarray) -> None:
        nonlocal best
        take = valid & (t > 0) & (t < best)
        best = np.where(take, t, best)
        label[take] = lab
        color[take] = rgb[take]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = -c[2] / d[..., 2]
        p = c + t[..., None] * d
        light = 0.85
        consider(t, p[..., 1] < 0, ROAD, sample(scene.asphalt, p[..., 0], p[..., 1]) * light)

        t = (s * c[1] - c[2]) / (d[..., 2] - s * d[..., 1])
        p = c + t[..., None] * d
        normal = np.array([0.0, -s, 1.0]) / np.hypot(s, 1.0)
        sun = np.array([-0.3, -0.5, 0.8]) / np.linalg.norm([-0.3, -0.5, 0.8])
        consider(t, (np.abs(p[..., 0]) <= hw) & (p[..., 1] >= 0) & (p[..., 1] <= length), RAMP, sample(scene.concrete, p[..., 0], p[..., 1]) * (0.55 + 0.45 * max(normal @ sun, 0)))

        t = (hc - c[2]) / d[..., 2]
        p = c + t[..., None] * d
        on_ramp = (np.abs(p[..., 0]) <= hw) & (p[..., 1] <= length)
        beyond = p[..., 1] >= scene.sidewalk_width
        consider(t, (p[..., 1] >= 0) & ~on_ramp & ~beyond, SIDEWALK, sample(scene.concrete, p[..., 0], p[..., 1]) * 0.95)
        consider(t, beyond, GRASS, sample(scene.grass, p[..., 0], p[..., 1]))

        for side in (-hw, hw):  # the vertical flanks of the ramp opening
            t = (side - c[0]) / d[..., 0]
            p = c + t[..., None] * d
            consider(t, (p[..., 1] >= 0) & (p[..., 1] <= length) & (p[..., 2] >= s * p[..., 1]) & (p[..., 2] <= hc), CURB, sample(scene.concrete, p[..., 1], p[..., 2]) * 0.7)

        t = -c[1] / d[..., 1]
        p = c + t[..., None] * d
        consider(t, (np.abs(p[..., 0]) > hw) & (p[..., 2] >= 0) & (p[..., 2] <= hc), CURB, sample(scene.concrete, p[..., 0], p[..., 2]) * 0.7)

    image = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    labels = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.clip(image, 0, 1), labels, rotation


def street_view_cameras(count: int, spacing: float, distance: float, height: float = 2.5) -> list[np.ndarray]:
    """Camera positions along the road, `distance` from the curb, evenly spread and centred on the ramp."""
    xs = (np.arange(count) - (count - 1) / 2) * spacing
    return [np.array([x, -distance, height]) for x in xs]


if __name__ == "__main__":
    out = __import__("pathlib").Path("_synth_preview")
    out.mkdir(exist_ok=True)
    scene = Scene(slope=0.08)
    views = []
    for i, position in enumerate(street_view_cameras(3, 5.0, 8.0)):
        image, labels, _ = render(scene, position, np.array([0.0, 0.8, 0.06]), 520, 518, 364)
        views.append((np.clip(image * 255, 0, 255).astype(np.uint8)))
        cv2.imwrite(str(out / f"view{i}.png"), cv2.cvtColor(views[-1], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"labels{i}.png"), (labels * 60).astype(np.uint8))
    cv2.imwrite(str(out / "montage.png"), cv2.cvtColor(np.hstack(views), cv2.COLOR_RGB2BGR))
    print("ramp length m:", scene.ramp_length)
