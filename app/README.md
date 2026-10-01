# LocationLabel desktop app

By Ikarus Delta. The logo lives in `assets/`; `gen_brand.py` prepares it from `..\logo\image.png`.

A Windows app that estimates ramp, driveway, curb and stop sign measurements from street-level photos.
Drop photos in, press Analyze, and get labeled images back. All values are approximations from a single photo.

## Using it

1. Under **What to find**, choose what you need: ramps, driveways, stop signs, poles, curbs. Everything is selected by default.
2. Press **Add photos**. A wizard shows an animation of where to stand and asks for the photos:
   - **Objects photo:** the photo with the most objects and the widest area (signs, poles, curbs, the street). Skipped if you only chose ramps and driveways.
   - **Ramp views:** three photos of the ramp or driveway from different positions along the street, about 30 degrees apart, with an animation for each. Skipped if you did not choose ramps or driveways. The button reads **Skip** until a photo is added and then **Next**, so any photo can be skipped (a slope needs at least 2).
   - **Review:** saves the photos in the layout the analysis reads (`image.<ext>` and `ramp/1.<ext>`, `ramp/2.<ext>` ...) under `%LOCALAPPDATA%\LocationLabel\inputs\<name>\`.
3. Pick where to save results, then press **Analyze**. A progress bar shows the stage; **Show details** shows the log.
4. The results page shows the labeled images. **Open folder** has the images and `result.json`.

The first analysis downloads the AI models (about 3 GB, once) and needs internet. The first ramp slope downloads one more model (about 4.7 GB, VGGT, which is licensed for non-commercial use unless you switch to Meta's commercial checkpoint; see the pipeline README). After that it works offline.

**Settings**
- **Use Google Street View** (off by default): fetch images from an address instead of using your own photos. Turning it on asks for a Google Maps API key. The key is stored in `%LOCALAPPDATA%\LocationLabel\settings.json` on your computer.
- **Your photos**: phone photo or Street View screenshot, optional camera height, and cropping of screenshot borders.

## Bulk analysis

The **Bulk** page analyzes many locations in one go.

1. Choose **What to find**, the **input folder** and a **results folder** (not the input folder, and not inside it).
2. The page checks the structure before anything runs and lists every folder with *Ready* or the problem. Only the structure is checked, not the names:

   ```
   input folder/
       any name/                one location
           any name.png         the photo for objects (needed for signs, poles, curbs)
           ramp/                3 or more photos of one ramp or driveway (needed for ramps, driveways)
               1.png  2.png  3.png
       another name/
           ...
   ```

   The only fixed name is `ramp` (any capitalization). Folders that don't fit are skipped, with the reason shown. Photos lying directly in the input folder, and folders starting with `.` or `_`, are ignored.
3. Press **Analyze N folders**. The folders are processed in groups of 8, so the models load once per group, and **Stop** finishes the current group and then stops.
4. Results use the same layout and the same names as the input, with a labeled image for every photo:

   ```
   results folder/
       any name/
           any name.jpg
           ramp/  1.jpg  2.jpg  3.jpg
           summary.json
       bulk_summary.csv         one row per folder: status, what was found, warnings
   ```

   **Skip folders that already have results** lets you rerun a big folder and only do what is missing.

## What is in this folder

| Path | What it is |
|---|---|
| `main.py` | entry point (`--selftest OUTDIR photo...` runs the pipeline without the window) |
| `gui/` | the window, settings page, and the job runner that turns errors into plain messages |
| `addr_To_coord/`, `cood_to_sview/` | stages 1 and 2: address to coordinates, Street View images (only used when Street View is on) |
| `user_images/` | stage 2 for your own photos |
| `label_objs/` | stage 3: finds ramps, driveways, signs, poles and curbs |
| `measure/` | stage 4: estimates measurements and draws the labels |
| `common/` | shared helpers; `settings.py` keeps all app data under `%LOCALAPPDATA%\LocationLabel` |
| `LocationLabel.spec`, `installer.iss`, `build.ps1` | build the exe and the installer |

## Releasing and updates

The installer is a small online installer. It downloads the program package named in `latest.txt`, checks its SHA-256, unpacks it and adds shortcuts. Installed apps read the same `latest.txt` and show a notice when a newer version is listed.

```
.\build.ps1 -BaseUrl "https://huggingface.co/USER/REPO/resolve/main" -Notes "What changed"
```

This builds the program (PyInstaller), zips it, and writes `release\`:
`LocationLabel-app-<version>.zip`, `latest.txt` and `LocationLabel.exe`. Upload all three to the repo (`hf upload USER/REPO release . --repo-type model`) and send people the small `LocationLabel.exe`.

- **New version:** raise `APP_VERSION` in `gui\version.py`, run the build again, upload the new zip and `latest.txt`. People on the old version see the update notice and install the new setup over it. Their settings and downloaded models are kept.
- **Only the URL changed:** run the same command with `-SkipApp`. It reuses the zip and takes seconds.

## Building

```
powershell -ExecutionPolicy Bypass -File build.ps1
```

This runs PyInstaller (creates `dist\LocationLabel\LocationLabel.exe`) and then Inno Setup (creates `installer\LocationLabel.exe`, with a standard install wizard).
Run it with the project's `env` activated so PySide6 and PyInstaller are available.
The build includes CUDA-enabled PyTorch, so it is several GB and works on computers with or without an NVIDIA card.

Logs are written to `%LOCALAPPDATA%\LocationLabel\logs\app.log`.
