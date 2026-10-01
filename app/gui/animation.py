"""A looping top-down animation that shows where to stand for each photo."""

from math import atan2, cos, degrees, radians, sin

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from . import style

# Where the camera stands for each ramp photo, in degrees from straight in front of the ramp (right is positive).
ANGLES = [0, 30, -30, 60, -60]
MOVE, HOLD, REST = 1.4, 1.3, 0.6
CYCLE = MOVE + HOLD + REST


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class RampAnimation(QWidget):
    """mode "ramp": the camera moves to the angle of the current photo. mode "overview": one wide shot of everything."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(230)
        self.step = 0
        self.mode = "ramp"
        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def set_step(self, index: int) -> None:
        self.step, self._t = min(max(index, 0), len(ANGLES) - 1), 0.0
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode, self._t = mode, 0.0
        self.update()

    def set_phase(self, seconds: float) -> None:
        """Freeze the animation at a moment in its cycle (used to render still frames)."""
        self._timer.stop()
        self._t = seconds
        self.update()

    def _tick(self) -> None:
        self._t = (self._t + 0.033) % CYCLE
        if self.isVisible():
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        area = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(style.BORDER), 1))
        painter.setBrush(QColor(style.PANEL))
        painter.drawRoundedRect(area, 16, 16)
        painter.setClipPath(self._rounded(area))

        curb_y = area.top() + 56
        centre = QPointF(area.center().x(), curb_y)
        self._draw_street(painter, area, curb_y)
        if self.mode == "overview":
            self._draw_overview(painter, area, centre)
        else:
            self._draw_ramp_shot(painter, area, centre)

    @staticmethod
    def _rounded(area: QRectF) -> QPainterPath:
        path = QPainterPath()
        path.addRoundedRect(area, 16, 16)
        return path

    def _draw_street(self, p: QPainter, area: QRectF, curb_y: float) -> None:
        p.fillRect(QRectF(area.left(), area.top(), area.width(), curb_y - area.top()), QColor("#272930"))
        p.fillRect(QRectF(area.left(), curb_y, area.width(), area.bottom() - curb_y), QColor("#1B1C20"))
        p.setPen(QPen(QColor("#3C3F48"), 2))
        p.drawLine(QPointF(area.left(), curb_y), QPointF(area.right(), curb_y))
        p.setPen(QPen(QColor("#2F3138"), 2, Qt.DashLine))
        p.drawLine(QPointF(area.left() + 10, area.bottom() - 14), QPointF(area.right() - 10, area.bottom() - 14))
        self._text(p, "sidewalk", QPointF(area.left() + 14, area.top() + 22), style.MUTED, 10)
        self._text(p, "road", QPointF(area.left() + 14, curb_y + 22), style.MUTED, 10)
        # the ramp: a strip cut through the curb, flared toward the road
        ramp = QPolygonF([QPointF(area.center().x() - 20, curb_y - 12), QPointF(area.center().x() + 20, curb_y - 12), QPointF(area.center().x() + 26, curb_y + 10), QPointF(area.center().x() - 26, curb_y + 10)])
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2F7F6C"))
        p.drawPolygon(ramp)
        self._text(p, "ramp", QPointF(area.center().x() + 34, curb_y - 2), "#7CC9B6", 10)

    def _camera_point(self, centre: QPointF, radius: float, angle: float) -> QPointF:
        return QPointF(centre.x() + radius * sin(radians(angle)), centre.y() + radius * cos(radians(angle)))

    def _draw_ramp_shot(self, p: QPainter, area: QRectF, centre: QPointF) -> None:
        radius = min(area.bottom() - centre.y() - 34, area.width() * 0.34)
        target = ANGLES[self.step]
        previous = ANGLES[self.step - 1] if self.step else 0
        moved = ease(self._t / MOVE)
        angle = previous + (target - previous) * moved if self.step else 0
        stand = radius * (1.0 + 0.25 * (1 - moved)) if self.step == 0 else radius  # photo 1: walk in to 3-8 m

        p.setPen(QPen(QColor("#33353D"), 1.5, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius), int((270 - 75) * 16), int(150 * 16))

        for done in range(self.step):
            spot = self._camera_point(centre, radius, ANGLES[done])
            self._badge(p, spot, str(done + 1), faded=True)

        position = self._camera_point(centre, stand, angle)
        if self.step:
            self._angle_arc(p, centre, target, angle)
        self._cone(p, position, centre, 26, style.ACCENT)
        self._camera_icon(p, position, centre)
        self._text(p, "3-8 m", QPointF((position.x() + centre.x()) / 2 + 8, (position.y() + centre.y()) / 2 + 6), style.BODY, 10)

        held = self._t - MOVE
        if 0 <= held <= HOLD:
            self._flash(p, position, held / HOLD)
            self._badge(p, QPointF(position.x() + 58, position.y()), str(self.step + 1), faded=False, pill=f"Photo {self.step + 1}")

    def _draw_overview(self, p: QPainter, area: QRectF, centre: QPointF) -> None:
        p.setPen(Qt.NoPen)
        objects = [(QPointF(area.center().x() - area.width() * 0.28, centre.y() - 26), "sign", "#D24B45"), (QPointF(area.center().x() + area.width() * 0.26, centre.y() - 26), "pole", "#C9A227")]
        for point, name, colour in objects:
            p.setBrush(QColor(colour))
            p.drawEllipse(point, 7, 7)
            self._text(p, name, QPointF(point.x() + 12, point.y() + 4), style.BODY, 10)
        curb_label = QPointF(area.center().x() - area.width() * 0.08, centre.y() + 30)
        self._text(p, "curb", curb_label, style.BODY, 10)

        radius = area.bottom() - centre.y() - 30
        moved = ease(self._t / MOVE)
        position = QPointF(centre.x(), centre.y() + radius * (1.1 - 0.1 * moved))
        self._cone(p, position, QPointF(centre.x(), centre.y() - 20), 50, style.ACCENT)
        self._camera_icon(p, position, centre)
        held = self._t - MOVE
        if 0 <= held <= HOLD:
            self._flash(p, position, held / HOLD)

    def _cone(self, p: QPainter, origin: QPointF, aim: QPointF, half_angle: float, colour: str) -> None:
        direction = atan2(aim.y() - origin.y(), aim.x() - origin.x())
        length = ((aim.x() - origin.x()) ** 2 + (aim.y() - origin.y()) ** 2) ** 0.5 * 1.02
        left = QPointF(origin.x() + length * cos(direction - radians(half_angle)), origin.y() + length * sin(direction - radians(half_angle)))
        right = QPointF(origin.x() + length * cos(direction + radians(half_angle)), origin.y() + length * sin(direction + radians(half_angle)))
        fill = QColor(colour)
        fill.setAlpha(46)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        p.drawPolygon(QPolygonF([origin, left, right]))
        edge = QColor(colour)
        edge.setAlpha(120)
        p.setPen(QPen(edge, 1.2))
        p.drawLine(origin, left)
        p.drawLine(origin, right)

    def _camera_icon(self, p: QPainter, at: QPointF, aim: QPointF) -> None:
        p.save()
        p.translate(at)
        p.rotate(degrees(atan2(aim.y() - at.y(), aim.x() - at.x())) + 90)
        p.setPen(QPen(QColor("white"), 1.5))
        p.setBrush(QColor(style.ACCENT))
        p.drawRoundedRect(QRectF(-11, -8, 22, 16), 4, 4)
        p.setBrush(QColor("white"))
        p.drawEllipse(QPointF(0, -10), 4, 3)
        p.restore()

    def _badge(self, p: QPainter, at: QPointF, number: str, faded: bool, pill: str | None = None) -> None:
        p.setPen(Qt.NoPen)
        colour = QColor(style.ACCENT if not faded else "#4B4E58")
        p.setBrush(colour)
        if pill:
            rect = QRectF(at.x() - 30, at.y() - 10, 60, 20)
            p.drawRoundedRect(rect, 10, 10)
            self._text(p, pill, QPointF(at.x() - 24, at.y() + 4), "white", 10, bold=True)
        else:
            p.drawEllipse(at, 10, 10)
            self._text(p, number, QPointF(at.x() - 3.5, at.y() + 4), "white", 10, bold=True)

    def _angle_arc(self, p: QPainter, centre: QPointF, target: float, current: float) -> None:
        """Wedge between the direction of photo 1 (straight ahead) and the current camera, with its size in degrees."""
        radius = 30
        rect = QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius)
        start = 270.0  # straight down from the ramp on screen
        p.setPen(QPen(QColor("#F0B429"), 2))
        p.setBrush(Qt.NoBrush)
        p.drawArc(rect, int(start * 16), int(current * 16))
        p.drawLine(centre, self._camera_point(centre, 46, 0))
        if abs(current) > 8:
            mid = current / 2
            spot = self._camera_point(centre, 56, mid)
            self._text(p, f"{abs(round(target))}°" if abs(current - target) < 1 else f"{abs(round(current))}°", QPointF(spot.x() - 10, spot.y() + 4), "#F0B429", 11, bold=True)

    def _flash(self, p: QPainter, at: QPointF, progress: float) -> None:
        ring = QColor("white")
        ring.setAlpha(int(200 * (1 - progress)))
        p.setPen(QPen(ring, 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(at, 14 + 30 * progress, 14 + 30 * progress)

    def _text(self, p: QPainter, text: str, at: QPointF, colour: str, size: int, bold: bool = False) -> None:
        font = QFont("Inter")
        font.setPixelSize(size + 2)
        font.setBold(bold)
        p.setFont(font)
        p.setPen(QColor(colour))
        p.drawText(at, text)
