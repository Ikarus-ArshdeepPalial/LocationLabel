"""The Bulk page: pick a folder of locations, check its structure, analyze them all, get results in the same layout."""

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from . import style
from .bulk import MIN_RAMP_PHOTOS, JobFolder, JobOutcome, scan_folder
from .icons import icon
from .page import Page
from .pipeline import GENERAL_TARGETS, SLOPE_TARGETS, TARGET_LABELS, friendly_error
from .settings_store import ALL_TARGETS, AppSettings, save_settings
from .widgets import Banner, ProgressBlock, ToggleSwitch, button, card, card_layout, chip_toggle, key_value, label
from .worker import BulkWorker

LAYOUT_EXAMPLE = """input folder/
    any name/                 one location
        any name.png          the photo for objects (signs, poles, curbs)
        ramp/                 photos of one ramp or driveway, 3 or more
            1.png  2.png  3.png
    another name/
        ..."""


def short(path: str, limit: int = 46) -> str:
    return path if len(path) <= limit else "..." + path[-(limit - 3):]


class BulkPage(Page):
    def __init__(self, settings: AppSettings, window) -> None:
        super().__init__("Bulk analysis")
        self._settings, self._window = settings, window
        self.jobs: list[JobFolder] = []
        self._worker: BulkWorker | None = None
        self._out_folder: Path | None = None

        self.start = button("Analyze", "primary", "scan")
        self.start.clicked.connect(self._start)
        self.add_actions(self.start)

        layout_card = card()
        layout_box = card_layout(layout_card, 8)
        layout_box.addWidget(label("Folder layout", "cardTitle"))
        example = label(LAYOUT_EXAMPLE, "value")
        example.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        layout_box.addWidget(example)
        layout_box.addWidget(label("Folder and photo names can be anything: only the structure is checked. The folder for the ramp photos must be called ramp. "
                                   "Results are written in the same layout, with a labeled image for every photo.", "muted", wrap=True))
        self.content.addWidget(layout_card)

        find = card()
        find_box = card_layout(find, 10)
        find_box.addWidget(label("What to find", "cardTitle"))
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.chips: dict[str, QPushButton] = {}
        for key, name in TARGET_LABELS.items():
            chip = chip_toggle(name)
            chip.setChecked(key in settings.bulk_targets)
            chip.toggled.connect(self._targets_changed)
            self.chips[key] = chip
            chips.addWidget(chip)
        chips.addStretch(1)
        find_box.addLayout(chips)
        self.find_hint = label("", "muted", wrap=True)
        find_box.addWidget(self.find_hint)
        self.content.addWidget(find)

        folders = card()
        folders_box = card_layout(folders, 8)
        folders_box.addWidget(label("Folders", "cardTitle"))
        self.input_label, self.output_label = label("", "value"), label("", "value")
        folders_box.addWidget(key_value("Input folder", self._path_row(self.input_label, self._choose_input)))
        folders_box.addWidget(key_value("Save results to", self._path_row(self.output_label, self._choose_output)))
        self.skip_done = ToggleSwitch(settings.bulk_skip_done)
        self.skip_done.toggled.connect(self._skip_changed)
        folders_box.addWidget(key_value("Skip folders that already have results", self.skip_done))
        self.content.addWidget(folders)

        self.check_card = card()
        check_box = card_layout(self.check_card, 10)
        head = QHBoxLayout()
        head.addWidget(label("Check", "cardTitle"))
        head.addStretch(1)
        recheck = button("Check again", "link")
        recheck.clicked.connect(self._scan)
        head.addWidget(recheck)
        check_box.addLayout(head)
        self.check_banner = Banner("note")
        check_box.addWidget(self.check_banner)
        self.loose_banner = Banner("warn")
        self.loose_banner.hide()
        check_box.addWidget(self.loose_banner)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Folder", "Objects photo", "Ramp photos", "Status"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setShowGrid(False)
        self.table.setWordWrap(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setStretchLastSection(True)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setHighlightSections(False)
        check_box.addWidget(self.table)
        self.check_card.hide()
        self.content.addWidget(self.check_card)

        self.error = Banner("error")
        self.error.hide()
        self.content.addWidget(self.error)

        self.run_card = card()
        run_box = card_layout(self.run_card)
        run_head = QHBoxLayout()
        run_head.addWidget(label("Analyzing", "cardTitle"))
        run_head.addStretch(1)
        self.stop = button("Stop", "small")
        self.stop.clicked.connect(self._stop)
        run_head.addWidget(self.stop)
        run_box.addLayout(run_head)
        self.progress = ProgressBlock()
        run_box.addWidget(self.progress)
        self.details_toggle = button("Show details", "link")
        self.details_toggle.clicked.connect(self._toggle_details)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.hide()
        run_box.addWidget(self.details_toggle, 0, Qt.AlignLeft)
        run_box.addWidget(self.log)
        self.run_card.hide()
        self.content.addWidget(self.run_card)

        self.done_card = card()
        done_box = card_layout(self.done_card, 10)
        self.done_banner = Banner("ok")
        done_box.addWidget(self.done_banner)
        open_row = QHBoxLayout()
        open_row.addStretch(1)
        self.open_results = button("Open results folder", "primary", "folder")
        self.open_results.clicked.connect(lambda: self._out_folder and QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._out_folder))))
        open_row.addWidget(self.open_results)
        done_box.addLayout(open_row)
        self.done_card.hide()
        self.content.addWidget(self.done_card)
        self.content.addStretch(1)

        self._refresh_paths()
        self._scan()

    # -- paths and choices ---------------------------------------------------------------------------------------

    def _path_row(self, text: QWidget, handler) -> QWidget:
        holder = QWidget()
        holder.setObjectName("transparent")
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        choose = button("Choose", "small", "folder", style.BODY)
        choose.clicked.connect(handler)
        row.addWidget(text)
        row.addWidget(choose)
        return holder

    def targets(self) -> tuple[str, ...]:
        chosen = tuple(key for key in ALL_TARGETS if key in self._settings.bulk_targets)
        return chosen or tuple(ALL_TARGETS)

    def _targets_changed(self) -> None:
        self._settings.bulk_targets = [key for key, chip in self.chips.items() if chip.isChecked()]
        save_settings(self._settings)
        self._scan()

    def _skip_changed(self, on: bool) -> None:
        self._settings.bulk_skip_done = on
        save_settings(self._settings)

    def _refresh_paths(self) -> None:
        s = self._settings
        self.input_label.setText(short(s.bulk_input) if s.bulk_input else "Not chosen")
        self.input_label.setToolTip(s.bulk_input)
        self.output_label.setText(short(s.bulk_output) if s.bulk_output else "Not chosen")
        self.output_label.setToolTip(s.bulk_output)

    def _choose_input(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "Choose the folder that holds one folder per location", self._settings.bulk_input or str(Path.home()))
        if name:
            self._settings.bulk_input = name
            save_settings(self._settings)
            self._refresh_paths()
            self._scan()

    def _choose_output(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "Choose where to save the results", self._settings.bulk_output or str(Path.home() / "Documents"))
        if name:
            self._settings.bulk_output = name
            save_settings(self._settings)
            self._refresh_paths()
            self._sync()

    # -- checking the structure ----------------------------------------------------------------------------------

    def _scan(self) -> None:
        targets = set(self.targets())
        needs = []
        if targets & GENERAL_TARGETS:
            needs.append("one photo directly in it")
        if targets & SLOPE_TARGETS:
            needs.append(f"a ramp folder with {MIN_RAMP_PHOTOS} or more photos")
        self.find_hint.setText("Each location folder needs " + " and ".join(needs) + "." if needs else "Select at least one thing to find.")
        self.error.hide()
        self.done_card.hide()
        root = Path(self._settings.bulk_input) if self._settings.bulk_input else None
        if root is None or not root.is_dir():
            self.jobs = []
            self.check_card.hide()
            self._sync()
            return
        scan = scan_folder(root, self.targets())
        self.jobs = scan.jobs
        self._fill_table()
        ready = sum(job.ok for job in self.jobs)
        if not self.jobs:
            self.check_banner.set_kind("error")
            self.check_banner.set_text("No folders were found inside the input folder. Each location needs its own folder.")
        elif ready == len(self.jobs):
            self.check_banner.set_kind("ok")
            self.check_banner.set_text(f"All {ready} folders have the right structure.")
        else:
            self.check_banner.set_kind("warn")
            self.check_banner.set_text(f"{ready} of {len(self.jobs)} folders are ready. The others have a problem (see below) and will be skipped.")
        self.loose_banner.setVisible(scan.loose_images > 0)
        if scan.loose_images:
            many = scan.loose_images != 1
            self.loose_banner.set_text(f"{scan.loose_images} photo{'s' if many else ''} {'lie' if many else 'lies'} directly in the input folder and {'belong' if many else 'belongs'} to no location, so {'they are' if many else 'it is'} ignored. Put {'each' if many else 'it'} in a folder of {'its' if not many else 'its'} own.")
        self.check_card.show()
        self._sync()

    def _fill_table(self) -> None:
        self.table.setRowCount(len(self.jobs))
        for row, job in enumerate(self.jobs):
            cells = [job.name, str(len(job.main)) if job.main else "none", str(len(job.ramp)) if job.ramp else "none"]
            for column, text in enumerate(cells):
                self.table.setItem(row, column, QTableWidgetItem(text))
            self._set_status(row, job.ok, "Ready" if job.ok else "; ".join(job.problems))
        self.table.resizeRowsToContents()
        content = self.table.horizontalHeader().height() + sum(self.table.rowHeight(r) for r in range(len(self.jobs))) + 6
        self.table.setFixedHeight(min(content, 640))

    def _set_status(self, row: int, ok: bool, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setIcon(icon("check" if ok else "alert", style.ACCENT if ok else style.AMBER, 18))
        item.setToolTip(text)
        self.table.setItem(row, 3, item)

    # -- running -------------------------------------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _sync(self) -> None:
        ready = [job for job in self.jobs if job.ok]
        running = self.is_running()
        self.start.setText(f"Analyze {len(ready)} folder{'s' if len(ready) != 1 else ''}" if ready else "Analyze")
        self.start.setEnabled(bool(ready) and bool(self._settings.bulk_output) and not running)
        for chip in self.chips.values():
            chip.setEnabled(not running)
        self.run_card.setVisible(running)

    def _start(self) -> None:
        self.error.hide()
        if self._window.is_busy():
            self._show_error("Another analysis is already running. Wait for it to finish.")
            return
        out_root = Path(self._settings.bulk_output)
        root = Path(self._settings.bulk_input)
        try:
            same = out_root.resolve() == root.resolve() or root.resolve() in out_root.resolve().parents
        except OSError:
            same = False
        if same:
            self._show_error("Choose a results folder that is not the input folder or inside it, so results never mix with your photos.")
            return
        ready = [job for job in self.jobs if job.ok]
        self._out_folder = out_root
        self.done_card.hide()
        self.log.clear()
        self._worker = BulkWorker(ready, self.targets(), self._settings, out_root, self.skip_done.isChecked())
        self._worker.progress.connect(self.progress.set_progress)
        self._worker.log.connect(self.log.appendPlainText)
        self._worker.finished_with.connect(self._finished)
        self._worker.failed.connect(self._failed)
        self.progress.set_progress(0.0, "Starting...")
        self._worker.start()
        self._sync()

    def _stop(self) -> None:
        if self._worker:
            self._worker.stop()
            self.stop.setEnabled(False)
            self.progress.stage.setText("Stopping after the folders being analyzed now...")

    def _finished(self, outcomes: list[JobOutcome]) -> None:
        self.stop.setEnabled(True)
        by_name = {o.name: o for o in outcomes}
        for row, job in enumerate(self.jobs):
            outcome = by_name.get(job.name)
            if outcome:
                ok = outcome.status in ("done", "skipped")
                detail = {"done": ", ".join(outcome.findings) or "Done, nothing found", "skipped": "Skipped: already has results"}.get(outcome.status, f"{outcome.status.capitalize()}: {outcome.message}")
                self._set_status(row, ok, detail)
        count = {kind: sum(o.status == kind for o in outcomes) for kind in ("done", "failed", "skipped", "stopped")}
        parts = [f"{count['done']} analyzed"] + [f"{count[k]} {k}" for k in ("failed", "skipped", "stopped") if count[k]]
        self.done_banner.set_kind("ok" if not count["failed"] and not count["stopped"] else "warn")
        self.done_banner.set_text("Finished: " + ", ".join(parts) + f". Results are in {self._out_folder}, with bulk_summary.csv listing every folder.")
        self.done_card.show()
        self._worker = None
        self._sync()

    def _failed(self, message: str) -> None:
        self.stop.setEnabled(True)
        self._worker = None
        self._show_error(message)
        self._sync()

    def _show_error(self, text: str) -> None:
        self.error.set_text(text)
        self.error.show()

    def _toggle_details(self) -> None:
        show = not self.log.isVisible()
        self.log.setVisible(show)
        self.details_toggle.setText("Hide details" if show else "Show details")
