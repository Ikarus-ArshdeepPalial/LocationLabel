import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox
from PySide6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import style
from .icons import icon, pixmap


def asset(name: str) -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "assets" / name


def label(text: str = "", name: str = "", wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def button(text: str, kind: str = "", icon_name: str = "", color: str = style.TEXT) -> QPushButton:
    widget = QPushButton(text)
    if kind:
        widget.setObjectName(kind)
    if icon_name:
        widget.setIcon(icon(icon_name, "white" if kind == "primary" else color, 18))
        widget.setIconSize(QSize(18, 18))
    widget.setCursor(Qt.PointingHandCursor)
    return widget


def chip_toggle(text: str) -> QPushButton:
    """A pill that can be switched on and off; on shows a check mark."""
    widget = QPushButton(text)
    widget.setObjectName("toggleChip")
    widget.setCheckable(True)
    widget.setCursor(Qt.PointingHandCursor)
    widget.setIcon(icon("check", style.FIELD, 14, checked_color=style.ACCENT))
    widget.setIconSize(QSize(14, 14))
    return widget


def card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def card_layout(frame: QFrame, spacing: int = 14) -> QVBoxLayout:
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(26, 22, 26, 22)
    layout.setSpacing(spacing)
    return layout


def chip_row(texts: list[str]) -> QWidget:
    holder = QWidget()
    holder.setObjectName("transparent")
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    for text in texts:
        row.addWidget(label(text, "chip"))
    row.addStretch(1)
    return holder


def key_value(name: str, control: QWidget | str) -> QWidget:
    """A muted label on the left and a value or control on the right, like the reference's detail rows."""
    holder = QWidget()
    holder.setObjectName("transparent")
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 4, 0, 4)
    row.addWidget(label(name, "muted"), 0, Qt.AlignVCenter)
    row.addStretch(1)
    row.addWidget(label(control, "value") if isinstance(control, str) else control, 0, Qt.AlignVCenter)
    return holder


def thumbnail(path: Path, size: int = 96) -> QIcon:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    image_size = reader.size()
    if image_size.isValid():
        reader.setScaledSize(image_size.scaled(size * 2, size * 2, Qt.KeepAspectRatio))
    image = reader.read()
    return QIcon(QPixmap.fromImage(image)) if not image.isNull() else QIcon()


class ToggleSwitch(QAbstractButton):
    def __init__(self, checked: bool = False) -> None:
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(44, 26)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style.ACCENT if self.isChecked() else "#3A3C44"))
        painter.drawRoundedRect(self.rect(), 13, 13)
        painter.setBrush(QColor("white"))
        painter.drawEllipse(self.width() - 23 if self.isChecked() else 3, 3, 20, 20)


class Banner(QFrame):
    """A tinted message box with an icon. kind: note, warn, error, ok, update."""

    ICONS = {"note": ("info", style.MUTED), "warn": ("alert", style.AMBER), "error": ("alert", style.RED), "ok": ("check", style.ACCENT), "update": ("download", style.ACCENT)}

    def __init__(self, kind: str, text: str = "") -> None:
        super().__init__()
        self.setObjectName(kind)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)
        name, color = self.ICONS[kind]
        mark = QLabel()
        mark.setPixmap(pixmap(name, color, 20))
        mark.setFixedSize(20, 20)
        self._mark = mark
        self._label = QLabel(text)
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(mark, 0, Qt.AlignTop)
        layout.addWidget(self._label, 1)

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def set_kind(self, kind: str) -> None:
        name, color = self.ICONS[kind]
        self._mark.setPixmap(pixmap(name, color, 20))
        self.setObjectName(kind)
        self.style().unpolish(self)
        self.style().polish(self)


class ProgressBlock(QWidget):
    """Big percentage, stage name, and a bar with 'done' on the left and 'left' on the right."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("transparent")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.percent = label("0%", "big")
        self.stage = label("", "bigSub", wrap=True)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        row = QHBoxLayout()
        self.done = label("0% done", "accentText")
        self.left = label("100% left", "mutedAccent")
        row.addWidget(self.done)
        row.addStretch(1)
        row.addWidget(self.left)
        for item in (self.percent, self.stage):
            layout.addWidget(item)
        layout.addSpacing(8)
        layout.addWidget(self.bar)
        layout.addLayout(row)

    def set_progress(self, fraction: float, text: str) -> None:
        pct = int(round(fraction * 100))
        self.bar.setValue(int(fraction * 1000))
        self.percent.setText(f"{pct}%")
        self.stage.setText(text)
        self.done.setText(f"{pct}% done")
        self.left.setText(f"{100 - pct}% left")


class BrandBadge(QWidget):
    """The Ikarus Delta logo with 'by Ikarus Delta' beside it."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("transparent")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        logo = QLabel()
        logo.setStyleSheet("background: #F4F4F6; border-radius: 10px;")
        logo.setContentsMargins(4, 4, 4, 4)
        picture = QPixmap(str(asset("ikarus_delta.png")))
        if not picture.isNull():
            logo.setPixmap(picture.scaledToHeight(44, Qt.SmoothTransformation))
        logo.setFixedSize(logo.sizeHint())
        row.addWidget(logo)
        row.addWidget(label("by Ikarus Delta", "muted"))


class DropZone(QFrame):
    """Accepts dropped photos and folders, or opens a file picker."""

    dropped = Signal(list)
    browse_files = Signal()
    browse_folder = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("dropzone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(190)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)

        mark = QLabel()
        mark.setPixmap(pixmap("upload", style.ACCENT, 34))
        mark.setAlignment(Qt.AlignCenter)
        title = label("Drag and drop photos here", "cardTitle")
        title.setAlignment(Qt.AlignCenter)
        hint = label("JPG or PNG photos of a ramp, driveway or stop sign, or a folder of them", "muted", wrap=True)
        hint.setAlignment(Qt.AlignCenter)

        buttons = QHBoxLayout()
        buttons.setAlignment(Qt.AlignCenter)
        photos = button("Choose photos", "small", "image", style.BODY)
        folder = button("Choose folder", "small", "folder", style.BODY)
        photos.clicked.connect(self.browse_files)
        folder.clicked.connect(self.browse_folder)
        buttons.addWidget(photos)
        buttons.addWidget(folder)

        for item in (mark, title, hint):
            layout.addWidget(item)
        layout.addSpacing(6)
        layout.addLayout(buttons)

    def _hover(self, on: bool) -> None:
        self.setProperty("hover", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._hover(True)

    def dragLeaveEvent(self, _event) -> None:
        self._hover(False)

    def dropEvent(self, event) -> None:
        self._hover(False)
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.dropped.emit(paths)


class NoWheelComboBox(QComboBox):
    """A dropdown that the mouse wheel does not change: scrolling over it scrolls the page instead."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()
