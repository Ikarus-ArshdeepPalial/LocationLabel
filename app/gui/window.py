import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFontDatabase, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import style
from .icons import icon
from .bulk_page import BulkPage
from .limits_page import LimitsPage
from .page import Page
from .pipeline import APPROXIMATE, GENERAL_TARGETS, SLOPE_TARGETS, TARGET_LABELS, TIPS, Inputs, JobResult
from .settings_store import ALL_TARGETS, SOURCE_TYPES, AppSettings, load_settings, save_settings
from .updates import UpdateInfo
from .version import APP_VERSION
from .widgets import (
    Banner,
    BrandBadge,
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    ProgressBlock,
    ToggleSwitch,
    asset,
    button,
    card,
    card_layout,
    chip_row,
    key_value,
    label,
    thumbnail,
)
from .wizard import PhotoWizard
from .worker import JobWorker, UpdateChecker

def clear(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()


class AnalyzePage(Page):
    def __init__(self, settings: AppSettings, window: "MainWindow") -> None:
        super().__init__("New analysis")
        self._settings, self._window = settings, window
        self.inputs = Inputs()
        self.folder: Path | None = None

        self.street_chip = button("", "dashed")
        self.street_chip.clicked.connect(lambda: window.goto(2))
        self.clear = button("Clear", "", "trash", style.BODY)
        self.clear.clicked.connect(self._clear)
        self.analyze = button("Analyze", "primary", "scan")
        self.analyze.clicked.connect(self._analyze)
        self.header.addSpacing(6)
        self.header.addWidget(self.street_chip)
        self.add_actions(self.clear, self.analyze)

        self.update_banner = Banner("update")
        self.update_button = button("Download", "small", "download")
        self.update_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self._update_url)))
        self._update_url = ""
        self.update_banner.layout().addWidget(self.update_button)
        self.update_banner.hide()
        self.content.addWidget(self.update_banner)

        self.run_card = card()
        self.content.addWidget(self.run_card)

        photos = card()
        photos_layout = card_layout(photos)
        photo_head = QHBoxLayout()
        photo_head.addWidget(label("Photos", "cardTitle"))
        photo_head.addStretch(1)
        self.count = label("", "muted")
        photo_head.addWidget(self.count)
        photos_layout.addLayout(photo_head)
        self.finding = label("", "muted", wrap=True)
        photos_layout.addWidget(self.finding)
        self.empty_hint = label("Press Add photos. You pick what to find first, then a short guide shows where to stand for each photo the analysis needs.", "muted", wrap=True)
        photos_layout.addWidget(self.empty_hint)
        self.thumbs = QListWidget()
        self.thumbs.setViewMode(QListWidget.IconMode)
        self.thumbs.setIconSize(QSize(96, 96))
        self.thumbs.setResizeMode(QListWidget.Adjust)
        self.thumbs.setMovement(QListWidget.Static)
        self.thumbs.setFixedHeight(146)
        self.thumbs.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbs.setSpacing(8)
        photos_layout.addWidget(self.thumbs)
        self.add_photos = button("Add photos", "primary", "upload")
        self.add_photos.clicked.connect(self._open_wizard)
        wizard_row = QHBoxLayout()
        wizard_row.addWidget(self.add_photos)
        wizard_row.addStretch(1)
        photos_layout.addLayout(wizard_row)
        self.content.addWidget(photos)

        details = card()
        details_layout = card_layout(details, 8)
        details_layout.addWidget(label("Details", "cardTitle"))
        details_layout.addSpacing(4)
        out_holder = QWidget()
        out_holder.setObjectName("transparent")
        out_row = QHBoxLayout(out_holder)
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.setSpacing(10)
        self.output = label("", "value")
        choose = button("Change", "small", "folder", style.BODY)
        choose.clicked.connect(self._choose_output)
        out_row.addWidget(self.output)
        out_row.addWidget(choose)
        details_layout.addWidget(key_value("Save results to", out_holder))
        self.photo_type = label("", "value")
        details_layout.addWidget(key_value("Photo type", self.photo_type))

        self.address_box = QWidget()
        self.address_box.setObjectName("transparent")
        address_layout = QVBoxLayout(self.address_box)
        address_layout.setContentsMargins(0, 8, 0, 0)
        address_layout.setSpacing(6)
        address_layout.addWidget(label("Address, to fetch Street View images when you have no photos", "muted"))
        self.address = QLineEdit()
        self.address.setPlaceholderText("2910 Sussex Gardens Ln, Austin, TX 78748")
        self.address.textChanged.connect(self._sync)
        address_layout.addWidget(self.address)
        details_layout.addWidget(self.address_box)
        self.content.addWidget(details)

        self.notice = Banner("warn")
        self.notice.hide()
        self.content.addWidget(self.notice)
        self.content.addWidget(Banner("note", APPROXIMATE + " " + TIPS))
        self.error = Banner("error")
        self.error.hide()
        self.content.addWidget(self.error)

        run_layout = card_layout(self.run_card)
        run_layout.addWidget(label("Analyzing", "cardTitle"))
        self.progress = ProgressBlock()
        run_layout.addWidget(self.progress)
        self.details_toggle = button("Show details", "link")
        self.details_toggle.clicked.connect(self._toggle_details)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.hide()
        run_layout.addWidget(self.details_toggle, 0, Qt.AlignLeft)
        run_layout.addWidget(self.log)
        self.run_card.hide()
        self.content.addStretch(1)

        self.apply_settings(settings)
        self._refresh()

    # -- what to find --------------------------------------------------------------------------------------------

    def targets(self) -> tuple[str, ...]:
        chosen = tuple(key for key in ALL_TARGETS if key in self._settings.targets)
        return chosen or tuple(ALL_TARGETS)  # a saved empty selection means everything, not nothing

    def apply_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        self.address_box.setVisible(settings.use_street_view)
        self.street_chip.setText("Street View on" if settings.use_street_view else "+ Street View")
        self._set_output(settings.output_dir)
        inverse = {v: k for k, v in SOURCE_TYPES.items()}
        self.photo_type.setText(inverse.get(settings.source_type, "Phone or camera photo"))
        self._sync()

    def _set_output(self, path: str) -> None:
        self.output.setText(path if len(path) <= 46 else "..." + path[-43:])
        self.output.setToolTip(path)

    def show_update(self, info: UpdateInfo) -> None:
        notes = f" {info.notes}" if info.notes else ""
        self.update_banner.set_text(f"Version {info.version} is available (you have {APP_VERSION}).{notes}")
        self._update_url = info.url
        self.update_button.setVisible(bool(info.url))
        self.update_banner.show()

    # -- photos --------------------------------------------------------------------------------------------------

    def _open_wizard(self) -> None:
        wizard = PhotoWizard(self.targets(), self.inputs if self.folder else None, self)
        if wizard.exec() == QDialog.Accepted:
            self._settings.targets = list(wizard.targets)
            save_settings(self._settings)
            self.inputs, self.folder = wizard.inputs, wizard.folder
            self.error.hide()
            self._refresh()

    def _refresh(self) -> None:
        self.thumbs.clear()
        for path, kind in [(p, "Objects") for p in self.inputs.main] + [(p, f"Ramp {i}") for i, p in enumerate(self.inputs.ramp, 1)]:
            item = QListWidgetItem(thumbnail(path), kind)
            item.setToolTip(str(path))
            item.setSizeHint(QSize(116, 130))
            self.thumbs.addItem(item)
        has = self.inputs.count > 0 and self.folder is not None
        self.thumbs.setVisible(has)
        self.empty_hint.setVisible(not has)
        self.add_photos.setText("Change photos" if has else "Add photos")
        self.count.setText(f"{len(self.inputs.main)} objects photo, {len(self.inputs.ramp)} ramp views" if has else "")
        self.finding.setText("Finding: " + ", ".join(TARGET_LABELS[t] for t in self.targets()))
        self._sync()

    def _sync(self) -> None:
        targets = set(self.targets())
        wants_slope, wants_general = bool(targets & SLOPE_TARGETS), bool(targets & GENERAL_TARGETS)
        typed = bool(self.address.text().strip())
        usable = self.folder is not None and (bool(self.inputs.main) or len(self.inputs.ramp) >= 2)
        ready = bool(targets) and (usable or (self._settings.use_street_view and typed))
        running = self.run_card.isVisible()
        self.analyze.setEnabled(ready and not running)
        self.clear.setEnabled((self.inputs.count > 0 or typed) and not running)
        self.add_photos.setEnabled(not running)

        problems = []
        if not targets:
            problems.append("Select at least one thing to find.")
        if self.folder is not None:
            if wants_general and not self.inputs.main:
                problems.append("Signs, poles or curbs are selected but there is no objects photo. Use Change photos to add one.")
            if wants_slope and len(self.inputs.ramp) < 2:
                problems.append("Ramps or driveways are selected without ramp views, so they get no reliable slope. Use Change photos to add them.")
        self.notice.set_text(" ".join(problems))
        self.notice.setVisible(bool(problems))

    def _choose_output(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "Choose where to save results", self._settings.output_dir)
        if name:
            self._settings.output_dir = name
            save_settings(self._settings)
            self._set_output(name)

    def _clear(self) -> None:
        self.inputs, self.folder = Inputs(), None
        self.address.clear()
        self.error.hide()
        self._refresh()

    def _toggle_details(self) -> None:
        show = not self.log.isVisible()
        self.log.setVisible(show)
        self.details_toggle.setText("Hide details" if show else "Show details")

    def _analyze(self) -> None:
        self.error.hide()
        self._window.start(self.folder, self.address.text(), self.targets())

    def show_error(self, message: str) -> None:
        self.error.set_text(message)
        self.error.show()

    def set_busy(self, busy: bool) -> None:
        self.run_card.setVisible(busy)
        if busy:
            self.log.clear()
            self.progress.set_progress(0.0, "Starting...")
        self._sync()

    def set_progress(self, fraction: float, text: str) -> None:
        self.progress.set_progress(fraction, text)

    def add_log(self, line: str) -> None:
        self.log.appendPlainText(line)


class ResultsPage(Page):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__("Results")
        self._pixmap = QPixmap()
        self._current: Path | None = None
        self._folder: Path | None = None

        open_folder = button("Open folder", "", "folder", style.BODY)
        open_folder.clicked.connect(lambda: self._folder and QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._folder))))
        again = button("New analysis", "primary", "plus")
        again.clicked.connect(lambda: window.goto(0))
        self.add_actions(open_folder, again)

        self.empty = card()
        empty_layout = card_layout(self.empty)
        empty_layout.addWidget(label("No results yet", "cardTitle"))
        empty_layout.addWidget(label("Analyze some photos and the labeled images will appear here.", "muted", wrap=True))
        self.content.addWidget(self.empty)

        self.summary = card()
        summary_layout = card_layout(self.summary, 10)
        self.total = label("", "big")
        self.total_label = label("", "bigSub")
        self.chips_holder = QVBoxLayout()
        summary_layout.addWidget(self.total)
        summary_layout.addWidget(self.total_label)
        summary_layout.addLayout(self.chips_holder)
        self.content.addWidget(self.summary)
        self.warnings = QVBoxLayout()
        self.content.addLayout(self.warnings)
        self.note = Banner("note", APPROXIMATE + " " + TIPS)
        self.content.addWidget(self.note)

        self.images_card = card()
        images_layout = card_layout(self.images_card)
        images_layout.addWidget(label("Labeled images", "cardTitle"))
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(360)
        self.preview.setCursor(Qt.PointingHandCursor)
        self.preview.setToolTip("Click to open the full-size image")
        self.preview.mousePressEvent = lambda _e: self._open_current()
        images_layout.addWidget(self.preview)
        self.strip = QListWidget()
        self.strip.setViewMode(QListWidget.IconMode)
        self.strip.setIconSize(QSize(96, 96))
        self.strip.setMovement(QListWidget.Static)
        self.strip.setFixedHeight(122)
        self.strip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.strip.setSpacing(8)
        self.strip.currentItemChanged.connect(self._select)
        images_layout.addWidget(self.strip)
        self.content.addWidget(self.images_card)
        self.content.addStretch(1)
        for widget in (self.summary, self.note, self.images_card):
            widget.hide()

    def show_result(self, result: JobResult) -> None:
        self._folder = result.output_folder
        self.empty.hide()
        for widget in (self.summary, self.note, self.images_card):
            widget.show()
        found = len(result.findings)
        self.total.setText(str(result.total))
        self.total_label.setText("object found" if result.total == 1 else "objects found")
        clear(self.chips_holder)
        if result.findings:
            self.chips_holder.addWidget(chip_row(result.findings))
        clear(self.warnings)
        for warning in result.warnings:
            self.warnings.addWidget(Banner("warn", warning))
        self.strip.clear()
        for path in result.images:
            item = QListWidgetItem(thumbnail(path), "")
            item.setData(Qt.UserRole, str(path))
            item.setSizeHint(QSize(112, 108))
            self.strip.addItem(item)
        if result.images:
            self.strip.setCurrentRow(0)
        self.strip.setVisible(len(result.images) > 1)

    def _select(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        self._current = Path(item.data(Qt.UserRole))
        self._pixmap = QPixmap(str(self._current))
        self._rescale()

    def _rescale(self) -> None:
        if not self._pixmap.isNull():
            self.preview.setPixmap(self._pixmap.scaled(max(self.preview.width(), 200), 560, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _open_current(self) -> None:
        if self._current:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current)))


class SettingsPage(Page):
    def __init__(self, settings: AppSettings, window: "MainWindow") -> None:
        super().__init__("Settings")
        self._settings = settings
        save = button("Save", "primary", "check")
        save.clicked.connect(lambda: window.goto(0))
        self.add_actions(save)

        street = card()
        street_layout = card_layout(street, 12)
        top = QHBoxLayout()
        text = QVBoxLayout()
        text.setSpacing(4)
        text.addWidget(label("Use Google Street View", "cardTitle"))
        text.addWidget(label("Fetch street images from an address when you have no photos. Off by default. It needs a Google Maps API key, and Google may charge for requests.", "muted", wrap=True))
        top.addLayout(text, 1)
        self.street = ToggleSwitch(settings.use_street_view)
        self.street.toggled.connect(self._street_toggled)
        top.addWidget(self.street, 0, Qt.AlignTop)
        street_layout.addLayout(top)
        self.key_box = QWidget()
        self.key_box.setObjectName("transparent")
        key_layout = QVBoxLayout(self.key_box)
        key_layout.setContentsMargins(0, 6, 0, 0)
        key_layout.setSpacing(8)
        key_layout.addWidget(label("Google Maps API key", "muted"))
        key_row = QHBoxLayout()
        self.key = QLineEdit(settings.api_key)
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Paste your API key")
        self.show_key = button("", "small")
        self.show_key.setCheckable(True)
        self.show_key.setIcon(icon("eye", style.BODY, 18))
        self.show_key.toggled.connect(self._toggle_key)
        key_row.addWidget(self.key, 1)
        key_row.addWidget(self.show_key)
        key_layout.addLayout(key_row)
        key_layout.addWidget(label("Turn on the Street View Static API and the Geocoding API for this key. It is stored only on this computer.", "muted", wrap=True))
        street_layout.addWidget(self.key_box)
        self.content.addWidget(street)

        photos = card()
        photo_layout = card_layout(photos, 10)
        photo_layout.addWidget(label("Your photos", "cardTitle"))
        self.source = NoWheelComboBox()
        self.source.addItems(list(SOURCE_TYPES))
        values = list(SOURCE_TYPES.values())
        self.source.setCurrentIndex(values.index(settings.source_type) if settings.source_type in values else 0)
        self.source.setMinimumWidth(260)
        photo_layout.addWidget(key_value("Photo type", self.source))
        self.height = NoWheelDoubleSpinBox()
        self.height.setRange(0.0, 4.0)
        self.height.setSingleStep(0.1)
        self.height.setDecimals(2)
        self.height.setSpecialValueText("Default")
        self.height.setValue(settings.camera_height or 0.0)
        self.height.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.height.setMinimumWidth(260)
        photo_layout.addWidget(key_value("Camera height above the road (m)", self.height))
        self.crop = ToggleSwitch(settings.crop_ui)
        photo_layout.addWidget(key_value("Crop screenshot borders", self.crop))
        self.content.addWidget(photos)

        updates = card()
        update_layout = card_layout(updates, 10)
        update_layout.addWidget(label("Updates", "cardTitle"))
        self.updates = ToggleSwitch(settings.check_updates)
        update_layout.addWidget(key_value("Tell me when a new version is available", self.updates))
        update_layout.addWidget(key_value("Installed version", APP_VERSION))
        self.content.addWidget(updates)
        self.content.addStretch(1)
        self.key_box.setVisible(settings.use_street_view)

    def _toggle_key(self, on: bool) -> None:
        self.key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password)
        self.show_key.setIcon(icon("eye-off" if on else "eye", style.BODY, 18))

    def _street_toggled(self, on: bool) -> None:
        if on and not self.key.text().strip():
            key, accepted = QInputDialog.getText(self, "Google Maps API key", "Paste your Google Maps API key to turn on Street View:", QLineEdit.Password)
            if not accepted or not key.strip():
                self.street.setChecked(False)
                return
            self.key.setText(key.strip())
        self.key_box.setVisible(on)

    def collect(self) -> AppSettings:
        s = self._settings
        s.use_street_view = self.street.isChecked() and bool(self.key.text().strip())
        s.api_key = self.key.text().strip()
        s.source_type = list(SOURCE_TYPES.values())[self.source.currentIndex()]
        s.camera_height = self.height.value() or None
        s.crop_ui = self.crop.isChecked()
        s.check_updates = self.updates.isChecked()
        return s


class MainWindow(QMainWindow):
    # (name, icon, index in the page stack). The stack order is fixed; the sidebar order is the order here.
    NAV = [("Analyze", "scan", 0), ("Bulk", "layers", 3), ("Results", "chart", 1), ("Limits", "ruler", 4), ("Settings", "sliders", 2)]
    NAMES = {0: "Analyze", 1: "Results", 2: "Settings", 3: "Bulk analysis", 4: "Limits"}

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LocationLabel")
        self.resize(1220, 820)
        self.setMinimumSize(1000, 680)
        self._settings = load_settings()
        self._worker: JobWorker | None = None
        self._current = 0

        self.analyze = AnalyzePage(self._settings, self)
        self.results = ResultsPage(self)
        self.settings_page = SettingsPage(self._settings, self)
        self.bulk = BulkPage(self._settings, self)
        self.limits = LimitsPage(self._settings)
        self.stack = QStackedWidget()
        for page in (self.analyze, self.results, self.settings_page, self.bulk, self.limits):
            self.stack.addWidget(page)

        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        shell = QFrame()
        shell.setObjectName("shell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(14, 14, 14, 14)
        shell_layout.setSpacing(12)
        shell_layout.addLayout(self._top_bar())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._sidebar())
        content = QFrame()
        content.setObjectName("panel")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.stack)
        body.addWidget(content, 1)
        shell_layout.addLayout(body, 1)
        outer.addWidget(shell)
        self.setCentralWidget(root)

        if self._settings.check_updates:
            self._checker = UpdateChecker()
            self._checker.found.connect(self.analyze.show_update)
            self._checker.start()

    def _top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(10)
        mark = QLabel()
        mark.setPixmap(QPixmap(str(asset("app.ico"))).scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        mark.setContentsMargins(6, 0, 6, 0)
        bar.addWidget(mark)
        self.pill = label("Analyze", "pill")
        bar.addWidget(self.pill)
        bar.addStretch(1)
        bar.addWidget(label(f"Version {APP_VERSION}", "versionChip"))
        return bar

    def _sidebar(self) -> QFrame:
        side = QFrame()
        side.setObjectName("panel")
        side.setFixedWidth(240)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(18, 24, 18, 20)
        layout.setSpacing(6)
        layout.addWidget(label("LocationLabel", "sideTitle"))
        layout.addWidget(label("Ramps, driveways and signs from photos", "muted", wrap=True))
        layout.addSpacing(18)
        self._nav = QButtonGroup(self)
        self._nav.setExclusive(True)
        for name, icon_name, index in self.NAV:
            nav = button(name, "nav")
            nav.setCheckable(True)
            nav.setIcon(icon(icon_name, style.MUTED, 18, checked_color=style.TEXT))
            nav.setIconSize(QSize(18, 18))
            nav.clicked.connect(lambda _checked, i=index: self.goto(i))
            self._nav.addButton(nav, index)
            layout.addWidget(nav)
        self._nav.button(0).setChecked(True)
        layout.addStretch(1)
        layout.addWidget(BrandBadge())
        return side

    def goto(self, index: int) -> None:
        if self._current == 2 and index != 2:
            save_settings(self.settings_page.collect())
            self.analyze.apply_settings(self._settings)
        self._current = index
        self.stack.setCurrentIndex(index)
        self._nav.button(index).setChecked(True)
        self.pill.setText(self.NAMES[index])

    def is_busy(self) -> bool:
        return bool(self._worker and self._worker.isRunning()) or self.bulk.is_running()

    def start(self, folder: Path | None, address: str, targets: tuple[str, ...]) -> None:
        if self.bulk.is_running():
            self.analyze.show_error("A bulk analysis is running. Wait for it to finish, or stop it on the Bulk page.")
            return
        output = Path(self._settings.output_dir)
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.analyze.show_error("The results folder cannot be created. Choose a different folder.")
            return
        self.analyze.set_busy(True)
        self._worker = JobWorker(folder, address, self._settings, output, targets)
        self._worker.progress.connect(self.analyze.set_progress)
        self._worker.log.connect(self.analyze.add_log)
        self._worker.succeeded.connect(self._done)
        self._worker.failed.connect(self._failed)
        self._worker.start()

    def _done(self, result: JobResult) -> None:
        self.analyze.set_busy(False)
        self.results.show_result(result)
        self.goto(1)

    def _failed(self, message: str) -> None:
        self.analyze.set_busy(False)
        self.analyze.show_error(message)

    def closeEvent(self, event) -> None:
        if self.is_busy():
            answer = QMessageBox.question(self, "Analysis in progress", "An analysis is still running. Quit anyway?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            for worker in (self._worker, self.bulk._worker):
                if worker and worker.isRunning():
                    worker.terminate()
        save_settings(self.settings_page.collect())
        event.accept()


def load_fonts() -> None:
    for weight in ("Regular", "Medium", "SemiBold"):
        QFontDatabase.addApplicationFont(str(asset(f"fonts/Inter-{weight}.ttf")))


def run_gui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    load_fonts()
    app.setStyleSheet(style.STYLESHEET)
    icon_file = asset("app.ico")
    if icon_file.exists():
        app.setWindowIcon(QIcon(str(icon_file)))
    window = MainWindow()
    window.show()
    return app.exec()
