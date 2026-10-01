# PyInstaller spec: builds dist/LocationLabel/LocationLabel.exe (a windowed app in a folder).
# Build with:  ..\env\Scripts\pyinstaller --noconfirm LocationLabel.spec

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# VGGT (ramp slope from several photos) is bundled as source in third_party/, not installed.
THIRD_PARTY = os.path.join(SPECPATH, "third_party", "vggt")
sys.path.insert(0, THIRD_PARTY)

datas, binaries, hiddenimports = [], [], []
for package in ("torchvision", "ultralytics", "geocalib", "timm", "safetensors", "kornia", "einops"):
    try:
        d, b, h = collect_all(package)
    except Exception:
        continue
    datas += d
    binaries += b
    hiddenimports += h

# transformers loads model code lazily by name, so the models this app uses must be listed.
for model in ("mask2former", "swin", "grounding_dino", "sam", "depth_anything", "dinov2", "bert", "auto", "detr", "resnet"):
    hiddenimports += collect_submodules(f"transformers.models.{model}")
hiddenimports += collect_submodules("transformers.integrations") + collect_submodules("transformers.image_processing_utils")
hiddenimports += collect_submodules("vggt")
hiddenimports += ["scipy.ndimage", "scipy.spatial", "scipy.special._cdflib"]

# Libraries that inspect their own source (torch.jit, timm, transformers) need the .py files next to the compiled ones.
collection_mode = {name: "pyz+py" for name in ("torch", "timm", "geocalib", "transformers", "kornia")}

a = Analysis(
    ["main.py"],
    pathex=[".", THIRD_PARTY],
    binaries=binaries,
    datas=datas + [("assets", "assets")],
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest", "IPython", "notebook", "PyQt5", "PyQt6", "PySide2", "PySide6.QtWebEngineCore", "PySide6.QtQuick", "PySide6.Qt3DCore", "PySide6.QtMultimedia"],
    module_collection_mode=collection_mode,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocationLabel",
    console=False,
    icon="assets/app.ico",
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="LocationLabel", upx=False)
