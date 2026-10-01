from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from .widgets import label


class Page(QWidget):
    """A title row with actions on the right, then a scrolling column of cards."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("transparent")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 26, 30, 0)
        outer.setSpacing(18)
        self.header = QHBoxLayout()
        self.header.setSpacing(10)
        self.header.addWidget(label(title, "pageTitle"))
        outer.addLayout(self.header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        holder = QWidget()
        holder.setObjectName("transparent")
        self.content = QVBoxLayout(holder)
        self.content.setContentsMargins(0, 0, 8, 26)
        self.content.setSpacing(16)
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)

    def add_actions(self, *widgets: QWidget) -> None:
        self.header.addStretch(1)
        for widget in widgets:
            self.header.addWidget(widget)
