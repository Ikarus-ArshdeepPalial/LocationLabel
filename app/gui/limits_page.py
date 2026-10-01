"""The Limits page: the limits to check measurements against, each with a margin that keeps near-misses quiet."""

from dataclasses import asdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget

from measure.limits import DEFAULT_MARGIN, DEFAULTS, MEASURES, SECTION_OF, SECTIONS, UNITS, Limit, from_base, to_base

from . import style
from .page import Page
from .settings_store import AppSettings, save_settings
from .widgets import Banner, NoWheelComboBox, NoWheelDoubleSpinBox, ToggleSwitch, button, card, card_layout, label


def spin(value: float, decimals: int = 2) -> NoWheelDoubleSpinBox:
    box = NoWheelDoubleSpinBox()
    box.setRange(0.0, 100000.0)
    box.setDecimals(decimals)
    box.setSingleStep(0.1)
    box.setButtonSymbols(NoWheelDoubleSpinBox.NoButtons)
    box.setValue(value)
    box.setMinimumWidth(86)
    return box


def measure_combo() -> NoWheelComboBox:
    """What a limit applies to, grouped under Sidewalk, Ramp, Driveway, Pole and Signs. The group names are headings, not choices."""
    combo = NoWheelComboBox()
    model = QStandardItemModel(combo)
    heading_font = QFont()
    heading_font.setBold(True)
    for section, items in SECTIONS.items():
        heading = QStandardItem(section)
        heading.setFlags(Qt.NoItemFlags)
        heading.setFont(heading_font)
        model.appendRow(heading)
        for key, text in items:
            item = QStandardItem(f"    {text}")
            item.setData(key, Qt.UserRole)
            model.appendRow(item)
    combo.setModel(model)
    return combo


def select_measure(combo: NoWheelComboBox, key: str) -> None:
    combo.setCurrentIndex(next((i for i in range(combo.count()) if combo.itemData(i) == key), 1))


class LimitRow(QFrame):
    changed = Signal()
    removed = Signal(object)

    def __init__(self, rule: Limit) -> None:
        super().__init__()
        self.setObjectName("card")
        self._unit = rule.unit
        grid = QGridLayout(self)
        grid.setContentsMargins(20, 16, 20, 16)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        self.enabled = ToggleSwitch(rule.enabled)
        self.name = QLineEdit(rule.name)
        self.name.setPlaceholderText("Name, for example Driveway slope (Austin)")
        remove = button("", "small", "trash", style.BODY)
        remove.setToolTip("Remove this limit")
        remove.clicked.connect(lambda: self.removed.emit(self))
        grid.addWidget(self.enabled, 0, 0)
        grid.addWidget(self.name, 0, 1)
        grid.addWidget(remove, 0, 2)

        self.measure = measure_combo()
        select_measure(self.measure, rule.measure)
        grid.addWidget(label("Applies to", "muted"), 1, 0)
        grid.addWidget(self.measure, 1, 1, 1, 2)

        self.limit = spin(rule.limit)
        self.unit = NoWheelComboBox()
        self.margin = spin(rule.margin)
        self.source = QLineEdit(rule.source)
        self.source.setPlaceholderText("Where it comes from, for example Austin Standard 433S-1A")
        self.margin_unit = label("", "muted")
        line = QHBoxLayout()
        line.setSpacing(8)
        for item in (label("Limit", "muted"), self.limit, self.unit):
            line.addWidget(item)
        line.addSpacing(16)
        for item in (label("Margin", "muted"), self.margin, self.margin_unit):
            line.addWidget(item)
        line.addStretch(1)
        holder = QWidget()
        holder.setObjectName("transparent")
        holder.setLayout(line)
        line.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(holder, 2, 1, 1, 2)
        grid.addWidget(label("Source", "muted"), 3, 0)
        grid.addWidget(self.source, 3, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, 70)

        self._fill_units(rule.unit)
        self.enabled.toggled.connect(self.changed)
        self.name.textEdited.connect(self.changed)
        self.source.textEdited.connect(self.changed)
        self.limit.valueChanged.connect(self.changed)
        self.margin.valueChanged.connect(self.changed)
        self.measure.currentIndexChanged.connect(self._measure_changed)
        self.unit.currentTextChanged.connect(self._unit_changed)

    def _family(self) -> str:
        return MEASURES[self.measure.currentData()][2]

    def _fill_units(self, wanted: str) -> None:
        self.unit.blockSignals(True)
        self.unit.clear()
        self.unit.addItems(UNITS[self._family()])
        self.unit.setCurrentText(wanted if wanted in UNITS[self._family()] else UNITS[self._family()][0])
        self.unit.blockSignals(False)
        self._unit = self.unit.currentText()
        self.margin_unit.setText(self._unit)

    def _measure_changed(self) -> None:
        if self._unit not in UNITS[self._family()]:  # slope to length or back: the old unit no longer fits
            self.limit.blockSignals(True)
            self.margin.blockSignals(True)
            self._fill_units("")
            self.margin.setValue(DEFAULT_MARGIN[self._unit])
            self.limit.blockSignals(False)
            self.margin.blockSignals(False)
        self.changed.emit()

    def _unit_changed(self, unit: str) -> None:
        """Keeps the same limit when the unit changes: 10 % becomes 5.71 degrees, 7 ft becomes 213.4 cm."""
        if not unit or unit == self._unit:
            return
        old_limit, old_margin = self.limit.value(), self.margin.value()
        new_limit = from_base(to_base(old_limit, self._unit), unit)
        new_margin = abs(from_base(to_base(old_limit + old_margin, self._unit), unit) - new_limit)
        self._unit = unit
        for box, value in ((self.limit, new_limit), (self.margin, new_margin)):
            box.blockSignals(True)
            box.setValue(round(value, 3))
            box.blockSignals(False)
        self.margin_unit.setText(unit)
        self.changed.emit()

    def section(self) -> str:
        return SECTION_OF.get(self.measure.currentData(), "Signs")

    def rule(self) -> Limit:
        return Limit(self.name.text().strip() or "Unnamed limit", self.measure.currentData(), self.limit.value(), self.unit.currentText(), self.margin.value(), self.source.text().strip(), self.enabled.isChecked())


class LimitsPage(Page):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__("Limits")
        self._settings = settings
        self.rows: list[LimitRow] = []
        self._headings: list = []

        reset = button("Restore defaults", "", "refresh", style.BODY)
        reset.clicked.connect(self._restore)
        add = button("Add limit", "primary", "plus")
        add.clicked.connect(self._add_new)
        self.add_actions(reset, add)

        intro = card()
        intro_box = card_layout(intro, 8)
        intro_box.addWidget(label("How limits work", "cardTitle"))
        intro_box.addWidget(label(
            "Add the limits that apply where you work. When a measured value goes past a limit by more than its margin, the labeled image shows the value "
            "and the limit in red. Going past a limit by less than the margin shows no message, so a result that is just over is not flagged. "
            "A limit marked 'at least' (such as the sign height) is flagged when the value is too low. Slopes can be in % or degrees.", "muted", wrap=True))
        intro_box.addWidget(label("Values come from photos and are estimates, not a compliance finding. Changes apply to the next analysis.", "muted", wrap=True))
        self.content.addWidget(intro)

        self.holder = QVBoxLayout()
        self.holder.setSpacing(12)
        self.content.addLayout(self.holder)
        self.empty = Banner("note", "No limits yet. Press Add limit, or Restore defaults.")
        self.content.addWidget(self.empty)
        self.content.addStretch(1)

        for entry in settings.limits:
            try:
                rule = Limit(**{k: entry[k] for k in ("name", "measure", "limit", "unit", "margin", "source", "enabled") if k in entry})
            except (TypeError, KeyError, ValueError):
                continue
            if rule.measure in MEASURES:  # a measure that no longer exists (such as the driveway cross slope) is dropped
                self._add_row(rule)
        self._sync()

    def _add_row(self, rule: Limit) -> LimitRow:
        row = LimitRow(rule)
        row.changed.connect(self._save)
        row.removed.connect(self._remove)
        self.rows.append(row)
        return row

    def _add_new(self) -> None:
        first = next(iter(MEASURES))
        row = self._add_row(Limit("New limit", first, 0.0, "%", DEFAULT_MARGIN["%"], "", True))
        row.name.setFocus()
        row.name.selectAll()
        self._save()

    def _remove(self, row: LimitRow) -> None:
        self.rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self._save()

    def _restore(self) -> None:
        for row in list(self.rows):
            row.setParent(None)
            row.deleteLater()
        self.rows.clear()
        for rule in DEFAULTS:
            self._add_row(Limit(**asdict(rule)))
        self._save()

    def _save(self) -> None:
        self._settings.limits = [asdict(row.rule()) for row in self.rows]
        save_settings(self._settings)
        self._sync()

    def _sync(self) -> None:
        self.empty.setVisible(not self.rows)
        self._arrange()

    def _arrange(self) -> None:
        """The list in sections (Sidewalk, Ramp, Driveway, Pole, Signs), each with a heading above its limits."""
        for heading in self._headings:
            heading.setParent(None)
            heading.deleteLater()
        self._headings.clear()
        while self.holder.count():
            self.holder.takeAt(0)
        order = list(SECTIONS)
        current = None
        for row in sorted(self.rows, key=lambda r: order.index(r.section())):
            if row.section() != current:
                current = row.section()
                heading = label(current, "cardTitle")
                self._headings.append(heading)
                self.holder.addWidget(heading)
            self.holder.addWidget(row)
