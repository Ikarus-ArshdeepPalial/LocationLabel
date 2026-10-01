"""The photo wizard: what to find, the photo used to find objects, then views of the ramp from different positions."""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from . import style
from .animation import ANGLES, RampAnimation
from .icons import icon
from .pipeline import GENERAL_TARGETS, IMAGE_SUFFIXES, INPUTS_DIR, MAX_RAMP_PHOTOS, MIN_RAMP_PHOTOS, SLOPE_TARGETS, TARGET_LABELS, Inputs, build_input_folder, collect_images, file_fingerprint
from .widgets import Banner, button, card, card_layout, chip_toggle, label, thumbnail

SMALL_WIDTH = 900

RAMP_TEXT = [
    ("Photo 1: face the ramp", "Stand about 3 to 8 m from the ramp, straight in front of it. Keep the whole ramp and some road in view."),
    ("Photo 2: step to the side", "Walk about 3 m sideways along the street, so you see the ramp from about 30° to one side. Keep the ramp in view."),
    ("Photo 3: the other side", "Now go to the other side of photo 1, about 30° the other way. Move your feet: turning on the spot gives no depth."),
    ("Photo 4 (optional): wider", "A wider angle helps a little: about 60° to the right of photo 1. You can skip this one."),
]
SELECT_TEXT = ("What do you want to find?", "Pick what you need. Everything is selected by default. The next steps only ask for the photos your choice needs.")
MAIN_TEXT = ("The photo with the most objects", "Choose the photo that shows the most objects and covers the most area: signs, poles, curbs and the street. This one is used to find them.")


class PhotoSlot(QFrame):
    """One photo: drop it here or click to choose. Shows a preview once chosen."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.path: Path | None = None
        self.setObjectName("dropzone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(240)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)
        self.preview = QLabel()
        self.preview.setFixedHeight(140)
        self.preview.setAlignment(Qt.AlignCenter)
        self.prompt = label("Drop a photo here", "cardTitle")
        self.prompt.setAlignment(Qt.AlignCenter)
        self.detail = label("", "muted")
        self.detail.setAlignment(Qt.AlignCenter)
        row = QHBoxLayout()
        row.setAlignment(Qt.AlignCenter)
        self.choose = button("Choose photo", "small", "image", style.BODY)
        self.choose.clicked.connect(self._pick)
        self.remove = button("Remove", "small", "trash", style.BODY)
        self.remove.clicked.connect(lambda: self.set_path(None))
        row.addWidget(self.choose)
        row.addWidget(self.remove)
        for widget in (self.preview, self.prompt, self.detail):
            layout.addWidget(widget)
        layout.addLayout(row)
        self._refresh()

    def set_path(self, path: Path | None) -> None:
        self.path = path
        self._refresh()
        self.changed.emit()

    def _refresh(self) -> None:
        has = self.path is not None
        self.preview.setVisible(has)
        self.remove.setVisible(has)
        self.choose.setText("Replace photo" if has else "Choose photo")
        if not has:
            self.prompt.setText("Drop a photo here")
            self.detail.setText("JPG or PNG")
            return
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)
        size = reader.size()
        image = QPixmap(str(self.path)) if not size.isValid() else QPixmap.fromImage(_scaled(reader, size))
        self.preview.setPixmap(image.scaledToHeight(140, Qt.SmoothTransformation))
        self.prompt.setText(self.path.name)
        self.detail.setText(f"{size.width()} x {size.height()} px" if size.isValid() else "")

    def _pick(self) -> None:
        patterns = " ".join(f"*{s}" for s in sorted(IMAGE_SUFFIXES))
        name, _ = QFileDialog.getOpenFileName(self, "Choose a photo", str(Path.home() / "Pictures"), f"Photos ({patterns})")
        if name:
            self.set_path(Path(name))

    def mousePressEvent(self, event) -> None:
        if self.path is None:
            self._pick()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        found = collect_images([Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()])
        if found:
            self.set_path(found[0])


def _scaled(reader: QImageReader, size: QSize):
    reader.setScaledSize(size.scaled(600, 600, Qt.KeepAspectRatio))
    return reader.read()


@dataclass
class Step:
    kind: str       # "main", "ramp" or "review"
    index: int = 0  # which ramp photo


def image_width(path: Path) -> int:
    reader = QImageReader(str(path))
    return reader.size().width() if reader.size().isValid() else 0


class StepDots(QWidget):
    """A row of numbered step markers: done, current and coming up."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("transparent")
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(12)

    def show_steps(self, names: list[str], current: int) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, name in enumerate(names):
            done, active = i < current, i == current
            colour = style.ACCENT if done else style.TEXT if active else style.MUTED
            mark = "✓ " if done else ""
            chip = label(f"{mark}{name}", "chip")
            chip.setStyleSheet(f"background: {style.ACCENT_SOFT if active or done else style.FIELD}; color: {colour}; border-radius: 8px; padding: 5px 10px; font-size: 12px;")
            self._row.addWidget(chip)
        self._row.addStretch(1)


class PhotoWizard(QDialog):
    """Steps: what to find, then the photo for objects (if needed), then the ramp views (if needed), then a review."""

    def __init__(self, targets: tuple[str, ...], existing: Inputs | None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add your photos")
        self.resize(880, 860)
        self.setMinimumSize(760, 680)
        self.setObjectName("wizard")
        self.setStyleSheet(f"QDialog#wizard {{ background: {style.SHELL}; }}")
        self.folder: Path | None = None
        self.inputs = Inputs()
        self.selected: set[str] = set(targets) or set(TARGET_LABELS)
        self.steps, self.names = self._plan()
        self.position = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(14)
        layout.addWidget(label("Add your photos", "pageTitle"))
        self.dots = StepDots()
        layout.addWidget(self.dots)

        self.animation = RampAnimation()
        layout.addWidget(self.animation)

        self.instructions = card()
        instruction_layout = card_layout(self.instructions, 6)
        self.title = label("", "cardTitle")
        self.text = label("", "muted", wrap=True)
        self.text.setMinimumHeight(44)
        instruction_layout.addWidget(self.title)
        instruction_layout.addWidget(self.text)
        layout.addWidget(self.instructions)

        self.find_card = card()
        find_layout = card_layout(self.find_card, 12)
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.chips: dict[str, QPushButton] = {}
        for key, name in TARGET_LABELS.items():
            chip = chip_toggle(name)
            chip.setChecked(key in self.selected)
            chip.toggled.connect(self._targets_changed)
            self.chips[key] = chip
            chips.addWidget(chip)
        chips.addStretch(1)
        find_layout.addLayout(chips)
        self.find_hint = label("", "muted", wrap=True)
        find_layout.addWidget(self.find_hint)
        layout.addWidget(self.find_card)

        self.main_slot = PhotoSlot()
        self.ramp_slots = [PhotoSlot() for _ in range(MAX_RAMP_PHOTOS)]
        for slot in [self.main_slot, *self.ramp_slots]:
            slot.changed.connect(self._sync)
            layout.addWidget(slot)
        self.notice = Banner("warn")
        self.notice.hide()
        layout.addWidget(self.notice)

        self.review = QListWidget()
        self.review.setViewMode(QListWidget.IconMode)
        self.review.setIconSize(QSize(96, 96))
        self.review.setMovement(QListWidget.Static)
        self.review.setFixedHeight(150)
        self.review.setSpacing(8)
        self.review.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.review)
        self.saved = label("", "muted", wrap=True)
        layout.addWidget(self.saved)
        layout.addStretch(1)

        footer = QHBoxLayout()
        self.cancel = button("Cancel")
        self.cancel.clicked.connect(self.reject)
        self.back = button("Back", "", "", style.BODY)
        self.back.clicked.connect(self._back)
        self.next = button("Next", "primary", "arrow-right")
        self.next.clicked.connect(self._next)
        for plain in (self.cancel, self.back):
            plain.setAutoDefault(False)
        self.next.setDefault(True)  # Enter means Next
        footer.addWidget(self.cancel)
        footer.addStretch(1)
        footer.addWidget(self.back)
        footer.addWidget(self.next)
        layout.addLayout(footer)

        if existing:
            self.main_slot.path = existing.main[0] if existing.main else None
            for slot, path in zip(self.ramp_slots, existing.ramp):
                slot.path = path
            for slot in [self.main_slot, *self.ramp_slots]:
                slot._refresh()
        self._show()

    # -- what is asked for -------------------------------------------------------------------------------------

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(key for key in TARGET_LABELS if key in self.selected)

    def _plan(self) -> tuple[list[Step], list[str]]:
        wants_general, wants_slope = bool(self.selected & GENERAL_TARGETS), bool(self.selected & SLOPE_TARGETS)
        steps = [Step("select")] + ([Step("main")] if wants_general else []) + ([Step("ramp", i) for i in range(MAX_RAMP_PHOTOS)] if wants_slope else []) + [Step("review")]
        names = ["What to find"] + (["Objects photo"] if wants_general else []) + (["Ramp views"] if wants_slope else []) + ["Review"]
        return steps, names

    def _targets_changed(self) -> None:
        self.selected = {key for key, chip in self.chips.items() if chip.isChecked()}
        self.steps, self.names = self._plan()
        self._show()

    # -- state -------------------------------------------------------------------------------------------------

    @property
    def step(self) -> Step:
        return self.steps[self.position]

    def _ramp_paths(self) -> list[Path]:
        return [s.path for s in self.ramp_slots if s.path is not None]

    def _have_photo(self) -> bool:
        step = self.step
        if step.kind == "select":
            return bool(self.selected)
        if step.kind == "main":
            return self.main_slot.path is not None
        if step.kind == "ramp":
            return self.ramp_slots[step.index].path is not None
        return True

    def _usable(self) -> bool:
        inputs = self._collect()
        return bool(inputs.main) or len(inputs.ramp) >= 2

    # -- navigation --------------------------------------------------------------------------------------------

    def _next(self) -> None:
        if self.step.kind == "review":
            self._finish()
            return
        self.position = min(self.position + 1, len(self.steps) - 1)
        self._show()

    def _back(self) -> None:
        self.position = max(self.position - 1, 0)
        self._show()

    def _show(self) -> None:
        step = self.step
        kind_name = {"select": "What to find", "main": "Objects photo", "ramp": "Ramp views", "review": "Review"}[step.kind]
        self.dots.show_steps(self.names, self.names.index(kind_name))
        self.find_card.setVisible(step.kind == "select")
        self.main_slot.setVisible(step.kind == "main")
        for i, slot in enumerate(self.ramp_slots):
            slot.setVisible(step.kind == "ramp" and i == step.index)
        self.review.setVisible(step.kind == "review")
        self.saved.setVisible(step.kind == "review")
        self.animation.setVisible(step.kind in ("main", "ramp"))
        if step.kind == "select":
            self.title.setText(SELECT_TEXT[0])
            self.text.setText(SELECT_TEXT[1])
            self.find_hint.setText(self._select_hint())
        elif step.kind == "main":
            self.animation.set_mode("overview")
            self.title.setText(MAIN_TEXT[0])
            self.text.setText(MAIN_TEXT[1])
        elif step.kind == "ramp":
            self.animation.set_mode("ramp")
            self.animation.set_step(step.index)
            title, text = RAMP_TEXT[step.index]
            self.title.setText(title)
            self.text.setText(f"{text}  ({abs(ANGLES[step.index])}° from photo 1)" if step.index else text)
        else:
            self._fill_review()
        self.back.setVisible(self.position > 0)
        self._sync()

    def _select_hint(self) -> str:
        if not self.selected:
            return "Select at least one thing to continue."
        asks = []
        if self.selected & GENERAL_TARGETS:
            asks.append("one photo with the most objects")
        if self.selected & SLOPE_TARGETS:
            asks.append(f"{MIN_RAMP_PHOTOS} photos of the ramp or driveway from different positions (a {MIN_RAMP_PHOTOS + 1}th is optional)")
        return "You will be asked for " + " and ".join(asks) + "."

    def _sync(self) -> None:
        """Sets the button for this step every time. Required steps show Next, disabled until there is a photo;
        only the optional 4th ramp photo shows Skip, which turns into Next once a photo is added."""
        step = self.step
        have = self._have_photo()
        if step.kind == "review":
            text, primary, enabled = "Save photos", True, self._usable()
        elif step.kind == "ramp" and step.index >= MIN_RAMP_PHOTOS:
            text, primary, enabled = ("Next", True, True) if have else ("Skip", False, True)
        else:
            text, primary, enabled = "Next", True, have
        self.next.setText(text)
        self.next.setEnabled(enabled)
        self.next.setObjectName("primary" if primary else "")
        self.next.setIcon(icon("arrow-right", "white" if primary else style.BODY, 18))
        self.next.style().unpolish(self.next)
        self.next.style().polish(self.next)
        self.notice.hide()
        if step.kind == "ramp" and have:
            self._check_ramp_photo(step.index)
        if step.kind == "review":
            self._review_warning()

    def _check_ramp_photo(self, index: int) -> None:
        path = self.ramp_slots[index].path
        earlier = [s.path for s in self.ramp_slots[:index] if s.path is not None]
        try:
            if any(file_fingerprint(path) == file_fingerprint(other) for other in earlier):
                self._warn("This is the same picture as an earlier photo. Each view needs to be taken from a new position.")
            elif 0 < image_width(path) < SMALL_WIDTH:
                self._warn(f"This photo is small ({image_width(path)} px wide). Larger photos give steadier measurements.")
        except OSError:
            pass

    def _warn(self, text: str) -> None:
        self.notice.set_text(text)
        self.notice.show()

    # -- review and saving -------------------------------------------------------------------------------------

    def _collect(self) -> Inputs:
        main = [self.main_slot.path] if self.main_slot.path and any(s.kind == "main" for s in self.steps) else []
        ramp = self._ramp_paths() if any(s.kind == "ramp" for s in self.steps) else []
        return Inputs(main, ramp)

    def _review_warning(self) -> None:
        inputs = self._collect()
        if len(inputs.ramp) == 1:
            self._warn("Only one ramp photo was added; at least 2 are needed to measure a slope.")
        elif not self._usable():
            self._warn("There are no photos to analyze yet. Go back and add some.")

    def _fill_review(self) -> None:
        inputs = self._collect()
        self.title.setText("Ready to save")
        self.text.setText(f"{len(inputs.main)} photo for objects and {len(inputs.ramp)} ramp views. They will be saved under {INPUTS_DIR} in the folder layout the analysis reads (image + ramp/).")
        self.review.clear()
        for path, kind in [(p, "Objects") for p in inputs.main] + [(p, f"Ramp {i}") for i, p in enumerate(inputs.ramp, 1)]:
            item = QListWidgetItem(thumbnail(path), kind)
            item.setSizeHint(QSize(116, 130))
            self.review.addItem(item)
        self.saved.setText("")

    def _finish(self) -> None:
        from datetime import datetime

        inputs = self._collect()
        missing = [p.name for p in inputs.main + inputs.ramp if not p.is_file()]
        if missing:  # a photo moved or deleted since it was chosen
            self._warn("These photos can no longer be found: " + ", ".join(missing) + ". Go back and choose them again.")
            return
        try:
            self.folder = build_input_folder(inputs, f"Analysis {datetime.now():%Y-%m-%d %H-%M-%S}")
        except Exception as exc:  # a failure here must be visible, not a button that silently does nothing
            self._warn(f"The photos could not be saved: {exc}")
            return
        self.inputs = inputs
        self.accept()
