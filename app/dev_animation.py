"""Development helper (not shipped): renders animation frames to PNG."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PySide6.QtWidgets import QApplication

from gui import style
from gui.animation import RampAnimation
from gui.window import load_fonts

out = Path(sys.argv[1] if len(sys.argv) > 1 else "_anim")
out.mkdir(exist_ok=True)
app = QApplication([])
load_fonts()
app.setStyleSheet(style.STYLESHEET)
widget = RampAnimation()
widget.resize(620, 240)
widget.show()
for name, mode, step, phase in [("overview", "overview", 0, 1.8), ("step1", "ramp", 0, 2.0), ("step2_moving", "ramp", 1, 0.7), ("step2_hold", "ramp", 1, 2.2), ("step3_hold", "ramp", 2, 2.2), ("step5_hold", "ramp", 4, 2.2)]:
    widget.set_mode(mode)
    widget.set_step(step)
    widget.set_phase(phase)
    app.processEvents()
    widget.grab().save(str(out / f"{name}.png"))
print("rendered")
