"""Development helper (not shipped): renders the pages and wizard steps offscreen to PNG so the layout can be checked."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
os.environ["LOCATIONLABEL_HOME"] = str(Path(__file__).parent / "_dev_home")

from PySide6.QtWidgets import QApplication

from gui import style
from gui.pipeline import Inputs, JobResult
from gui.updates import UpdateInfo
from gui.window import MainWindow, load_fonts
from gui.wizard import PhotoWizard

out = Path(sys.argv[1] if len(sys.argv) > 1 else "_shots")
out.mkdir(exist_ok=True)
photos = [Path(p) for p in sys.argv[2:]]
app = QApplication([])
load_fonts()
app.setStyleSheet(style.STYLESHEET)


def shot(widget, name: str) -> None:
    app.processEvents()
    widget.grab().save(str(out / f"{name}.png"))


window = MainWindow()
window.show()
shot(window, "analyze_empty")

# the wizard, step by step
wizard = PhotoWizard(("ramps", "driveways", "signs", "poles", "curbs"), None, window)
wizard.show()
wizard.animation.set_phase(2.0)
shot(wizard, "wizard1_main")
wizard.main_slot.set_path(photos[0])
wizard._next()
wizard.animation.set_phase(2.0)
shot(wizard, "wizard2_ramp1")
wizard.ramp_slots[0].set_path(photos[0])
wizard._next()
wizard.animation.set_phase(2.2)
wizard.ramp_slots[1].set_path(photos[1])
shot(wizard, "wizard3_ramp2")
wizard._next()
wizard.ramp_slots[2].set_path(photos[2])
wizard.animation.set_phase(2.2)
shot(wizard, "wizard4_ramp3")
wizard.position = len(wizard.steps) - 1
wizard._show()
shot(wizard, "wizard5_review")
wizard.reject()

# the analyze page with photos, and the other pages
window.analyze.inputs = Inputs([photos[0]], [photos[0], photos[1], photos[2]])
window.analyze.folder = Path(".")
window.analyze._refresh()
shot(window, "analyze_ready")

window.analyze.show_update(UpdateInfo("1.1.0", "https://example.com", "Better driveway outlines."))
window.analyze.set_busy(True)
window.analyze.set_progress(0.42, "Finding what you selected (the first run downloads models)")
window.analyze.add_log("12:00:01  Loading segmentation model")
window.analyze.details_toggle.click()
shot(window, "analyze_running")
window.analyze.set_busy(False)

window.goto(1)
window.results.show_result(JobResult(Path("."), photos, ["1 driveway found", "Ramp slope about 4.3° (confidence 0.41 of 1)"], ["Nothing could be measured with enough confidence."], 1))
shot(window, "results")
window.settings_page.key.setText("AIza-example")
window.settings_page.street.setChecked(True)
window.goto(2)
shot(window, "settings")
window.settings_page.street.blockSignals(True)
window.settings_page.street.setChecked(False)
window.settings_page.key.setText("")
window.goto(0)
print("rendered")
