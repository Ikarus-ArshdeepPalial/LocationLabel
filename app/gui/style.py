"""Dark theme in the structure of the reference design: soft panels, rounded white-on-dark cards, one green accent."""

OUTER = "#0B0C0E"
SHELL = "#121316"
PANEL = "#17181C"
CARD = "#1F2025"
CARD_HOVER = "#25272D"
FIELD = "#2A2B31"
BORDER = "#2C2E35"
DASHED = "#3C3F48"
TEXT = "#F2F2F4"
BODY = "#C9CAD0"
MUTED = "#8E9099"
ACCENT = "#14A37F"
ACCENT_HOVER = "#19B58F"
ACCENT_SOFT = "#173A32"
TRACK = "#22332E"
AMBER = "#E8B923"
RED = "#E5675C"
FONT = '"Inter", "Segoe UI Variable Text", "Segoe UI", sans-serif'

STYLESHEET = f"""
* {{ font-family: {FONT}; font-size: 13px; color: {BODY}; }}
QMainWindow, QWidget#root {{ background: {OUTER}; }}
QFrame#shell {{ background: {SHELL}; border: 1px solid {BORDER}; border-radius: 24px; }}
QFrame#panel {{ background: {PANEL}; border-radius: 20px; }}
QWidget#transparent, QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollArea {{ border: none; }}

QLabel {{ background: transparent; }}
QLabel#pageTitle {{ font-size: 22px; font-weight: 500; color: {TEXT}; }}
QLabel#sideTitle {{ font-size: 18px; font-weight: 500; color: {TEXT}; }}
QLabel#cardTitle {{ font-size: 15px; font-weight: 500; color: {TEXT}; }}
QLabel#big {{ font-size: 30px; font-weight: 500; color: {TEXT}; }}
QLabel#bigSub {{ font-size: 14px; color: {TEXT}; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#value {{ color: {TEXT}; }}
QLabel#accentText {{ color: {ACCENT}; font-weight: 600; font-size: 12px; }}
QLabel#mutedAccent {{ color: #5E9F8F; font-size: 12px; }}
QLabel#chip {{ background: {FIELD}; border-radius: 8px; padding: 5px 10px; color: {BODY}; font-size: 12px; }}
QLabel#pill {{ background: {CARD}; border-radius: 12px; padding: 9px 16px; color: {TEXT}; }}
QLabel#versionChip {{ background: {CARD}; border-radius: 10px; padding: 6px 12px; color: {MUTED}; font-size: 12px; }}

QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 20px; }}
QFrame#dropzone {{ background: transparent; border: 1.5px dashed {DASHED}; border-radius: 16px; }}
QFrame#dropzone[hover="true"] {{ border-color: {ACCENT}; background: {ACCENT_SOFT}; }}
QFrame#note {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#warn {{ background: #2B2414; border: 1px solid #4A3D18; border-radius: 14px; }}
QFrame#error {{ background: #2E1A18; border: 1px solid #55302C; border-radius: 14px; }}
QFrame#ok {{ background: {ACCENT_SOFT}; border: 1px solid #1F5648; border-radius: 14px; }}
QFrame#update {{ background: {ACCENT_SOFT}; border: 1px solid #1F5648; border-radius: 14px; }}

QPushButton {{ background: {FIELD}; border: none; border-radius: 12px; padding: 10px 18px; color: {TEXT}; font-weight: 500; }}
QPushButton:hover {{ background: #33353D; }}
QPushButton:disabled {{ background: #1E1F24; color: #5C5E67; }}
QPushButton#primary {{ background: {ACCENT}; color: white; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background: #17332B; color: #5F8F83; }}
QPushButton#dashed {{ background: transparent; border: 1.5px dashed {DASHED}; color: {BODY}; padding: 6px 12px; border-radius: 10px; }}
QPushButton#dashed:hover {{ border-color: {ACCENT}; color: {TEXT}; }}
QPushButton#link {{ background: transparent; color: {ACCENT}; padding: 4px 6px; }}
QPushButton#link:hover {{ color: {ACCENT_HOVER}; }}
QPushButton#small {{ padding: 6px 12px; border-radius: 10px; }}
QPushButton#toggleChip {{ background: {FIELD}; color: {MUTED}; border: 1.5px solid transparent; border-radius: 10px; padding: 8px 14px 8px 10px; }}
QPushButton#toggleChip:hover {{ color: {TEXT}; }}
QPushButton#toggleChip:checked {{ background: {ACCENT_SOFT}; color: {TEXT}; border: 1.5px solid {ACCENT}; }}
QPushButton#nav {{ background: transparent; color: {MUTED}; text-align: left; padding: 11px 14px; border-radius: 12px; font-weight: 500; }}
QPushButton#nav:hover {{ background: {CARD}; color: {TEXT}; }}
QPushButton#nav:checked {{ background: {CARD_HOVER}; color: {TEXT}; }}

QLineEdit, QComboBox, QDoubleSpinBox {{ background: {FIELD}; border: 1px solid transparent; border-radius: 10px; padding: 8px 12px; color: {TEXT}; min-height: 20px; selection-background-color: {ACCENT}; }}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QLineEdit:read-only {{ color: {BODY}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox QAbstractItemView {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; selection-background-color: {FIELD}; outline: none; padding: 4px; }}

QProgressBar {{ background: {TRACK}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ background: {FIELD}; border: 1px solid transparent; border-radius: 12px; padding: 4px; color: {BODY}; }}
QListWidget::item:selected {{ border: 1.5px solid {ACCENT}; color: {TEXT}; background: {FIELD}; }}
QPlainTextEdit {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; font-family: Consolas, monospace; font-size: 12px; color: {BODY}; padding: 6px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #34363E; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #454852; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ background: none; height: 0; width: 0; }}
QScrollBar:horizontal {{ height: 0; }}

QTableWidget {{ background: transparent; border: none; gridline-color: transparent; color: {BODY}; outline: none; }}
QTableWidget::item {{ padding: 6px 8px; border-bottom: 1px solid {BORDER}; }}
QTableWidget::item:selected {{ background: {FIELD}; color: {TEXT}; }}
QHeaderView {{ background: transparent; border: none; }}
QTableCornerButton::section {{ background: transparent; border: none; }}
QHeaderView::section {{ background: transparent; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER}; padding: 8px; text-align: left; font-weight: 500; }}
QToolTip {{ background: {CARD_HOVER}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px 8px; }}
QMessageBox, QInputDialog, QFileDialog {{ background: {CARD}; }}
"""
