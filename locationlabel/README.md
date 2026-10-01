# LocationLabel

This pipeline analyzes US residential properties from Street View imagery. Each stage is a package, and all stages share one `.env`, one `requirements.txt` and one `main.py`.

| Stage | Package | Adds to `output/<address_slug>/result.json` |
|---|---|---|
| 1. Geocoding | `addr_To_coord` | top-level fields (`lat`, `lng`, `provider`, `match_quality`, `street_segment`, …) |
| 2. Street View | `cood_to_sview` | `streetview` |
| 2′. Your own images (instead of 2) | `user_images` | `streetview` with `"source": "user_supplied"` |
| 3. Object labeling | `label_objs` | `detections` |
| 4. Measurement | `measure` | `measurements` |
| shared helpers | `common` | record I/O, SQLite cache, HTTP session, CSV reading |

Each stage adds its own keys to the record and leaves the others untouched. The stages find each other's data through the record, so they must share one `OUTPUT_DIR`.

## Setup

```bash
pip install -r requirements.txt     # installs the CUDA build of torch (cu126); see the comment in the file
copy .env.example .env              # set GOOGLE_MAPS_API_KEY
```

Stages 1 and 2 need only `requests`, `python-dotenv` and `Pillow`. The ML packages are imported only when `detect` runs.

## Usage

```bash
python main.py geocode    "2910 Sussex Gardens Ln, Austin, TX 78748, USA"
python main.py streetview "2910 Sussex Gardens Ln, Austin, TX 78748, USA" --dry-run
python main.py streetview "2910 Sussex Gardens Ln, Austin, TX 78748, USA"
python main.py detect     "2910 Sussex Gardens Ln, Austin, TX 78748, USA"
python main.py measure    "2910 Sussex Gardens Ln, Austin, TX 78748, USA"
python main.py all        "2910 Sussex Gardens Ln, Austin, TX 78748, USA"

python main.py all --csv sample_addresses.csv
python main.py evaluate ground_truth.csv
```

The CSV needs an `address` column. Use the same address text for every stage, because the record folder name is derived from it.

| Command | Flags |
|---|---|
| all stages | `--csv FILE`, `--force` (ignore caches), global `--log-level` (before the stage name) |
| `streetview` | `--dry-run` (free metadata calls only), `--refine-with-google` |
| `detect` | `--model <hf id>` |
| `measure` | `--export-pointcloud` (.ply per image, coloured by class) |

Caches and YOLO weights live in `.cache/`. Hugging Face weights live in the HF cache (`HF_HOME`).

## Using your own images

```bash
python main.py all "2910 Sussex Gardens Ln, Austin, TX 78748, USA" --images-dir D:\photos\sussex
python main.py all "..." --images-dir D:\shots --source-type google_screenshot --crop-ui
python main.py all --csv batch.csv          # rows with an images_dir column use your images; blank rows use Street View
```

With `--images-dir`, Stage 2 makes no Google calls and needs no API key. Stage 1 still geocodes the address.

The images (any size, EXIF orientation applied) are copied to `output/<slug>/images/`, and the `streetview` record is written in the same shape Stage 2 produces. Stages 3 and 4 work on them unchanged.

**Ramp slope from several photos.** One photo can't give a reliable slope: in a synthetic test with known slopes, the single-photo depth method's estimates did not follow the true slope (correlation about 0), while three photos gave about ±0.6°. So put photos of the ramp taken at different positions in a `ramp` subfolder:

```
images/sussex/
    image.png          the ordinary image: stop signs, poles, curbs and anything else
    ramp/
        1.png          3 to 6 photos of the same ramp or driveway,
        2.png          taken from different positions along the street
        3.png
```

- Photos in `ramp/` are measured together, and only they give the ramp slope. Ordinary images no longer get a single-photo slope when a `ramp/` folder exists. Without one, the old single-photo estimate is used and is a rough guess.
- **Take them from different positions, not by turning in place.** Turning on the spot gives no depth. Neighbouring photos should be about 30 to 60° apart as seen from the ramp (roughly 3 m sideways at 6 to 10 m distance). Very wide changes (about 100° or more) were about twice as inaccurate in the test.
- Keep the ramp and some road in every photo, and about 3 to 8 m away.
- Stage 3 finds the ramp in each photo; Stage 4 rebuilds the scene with VGGT, fits planes to the ramp and the road, and reports the ramp's slope **relative to the road plane**. Confidence comes from how well the views agree with each other (leave-one-out).
- The result is drawn on the middle photo only (the 2nd of 3, the 2nd of 4, the 3rd of 5; photos are ordered by file name), because it is the likeliest to face the ramp most directly: `output/<slug>/measured/ramp_<name>.jpg`. It is also stored in `measurements.ramp` (with `"method": "multiview"`).
- **Licence:** VGGT's original checkpoint (`VGGT_MODEL`, default `facebook/VGGT-1B`, about 4.7 GB) is for non-commercial use. For a product, apply for `facebook/VGGT-1B-Commercial` and set `VGGT_MODEL`. The code is bundled unmodified in `third_party/vggt/` with its licence.
- The tested accuracy is for ideal synthetic scenes with perfect masks; expect real photos to be worse. Check it against a few ramps you have measured with a level.

**Where camera parameters come from.** Each field uses the first available source, and that source is recorded in `metadata_sources`:

1. `--camera-height` (camera height only)
2. `<image_stem>.json` next to the image, then `camera.json` in the folder (for all images). Keys: `fov_deg` or `focal_px`, `pitch_deg`, `roll_deg`, `heading_deg`, `camera_height_m`, `lat`, `lng`. See `examples/`.
3. EXIF: 35 mm-equivalent focal length → focal length in pixels, GPS → position, `GPSImgDirection` → heading.
4. Estimated from the image: first GeoCalib (focal length, pitch, roll; CPU or GPU, 116 MB of weights), then vanishing points of vertical lines (RANSAC).
5. Defaults per `--source-type` (`user_images/config.py`):
   - `phone`: 1.5 m height, 68° FOV
   - `google_screenshot`: 2.5 m height, 90° FOV

A detected roll is removed by rotating the image upright. `--crop-ui` removes flat UI bars plus the configured margins, and shifts the principal point to match.

**Effects on Stage 4:**
- Estimates that depend on `estimated` or `default` parameters get a lower confidence (see `METHOD_DEPENDENCIES` in `measure/config.py`) and a smaller weight in the fused median.
- The reference-scale method depends least on the camera (the focal length cancels out), so it dominates when metadata is guessed.
- Images without a known position and heading are not triangulated, and the reason is recorded.

## Stage 1 — geocoding

### Providers

| Provider | When used | `match_quality` |
|---|---|---|
| Google Geocoding | `GOOGLE_MAPS_API_KEY` is set | `ROOFTOP`, `RANGE_INTERPOLATED`, `GEOMETRIC_CENTER`, `APPROXIMATE` (+ `partial_match`) |
| US Census | no key, Google error, or no Google result | always `RANGE_INTERPOLATED` (interpolated along TIGER street ranges) |

Only `ROOFTOP` coordinates are actually on the house. Anything else is an approximate point on or near the street.

### Result record

`result.json` is a shared record for the whole pipeline. This stage writes the keys below.
Later stages add their own keys (`streetview`, `detections`, `measurements`), and re-running this stage keeps them.

```json
{
  "input_address": "...",
  "formatted_address": "...",
  "lat": 38.8987,
  "lng": -77.0352,
  "provider": "google | census",
  "match_quality": "ROOFTOP",
  "partial_match": false,
  "status": "ok | error",
  "error": null,
  "timestamp": "ISO-8601"
}
```


## Stage 2 — Street View

### How it works

1. **Target point:** Census points are interpolated on the street, so the target is moved `side_offset_m` (15 m) perpendicular to the street toward the TIGER side (L/R). This uses `street_segment` from Stage 1.
2. **Panoramas:** the metadata endpoint (free, `source=outdoor`) finds the main pano. It searches at 50, then 100, then 200 m. It also looks for up to 2 neighbor panos `neighbor_step_m` (15 m) along the street in each direction.
3. **Views:** the headings are initial great-circle bearings from each pano to the target. The view list is `MAIN_VIEWS` / `NEIGHBOR_VIEWS` in `config.py`.
4. **Images:** saved to `images/`, with a `contact_sheet.jpg` for visual review. Existing files are reused unless `--force` is passed.

Each entry in `streetview.images` includes the pano position, heading, pitch, FOV, size, the distance to the target, and the camera intrinsics
(`fx = fy = width / (2·tan(fov/2))`, `cx = width/2`, `cy = height/2`).

`streetview.status` is one of `ok`, `partial` (some views failed), `no_imagery`, or `error`.

Only the provider is Google-specific. To add another source (for example Mapillary), subclass `StreetViewProvider` in `provider.py`.


## Stage 3 — object labeling

### Models and downloaded weights

| Model | Role | Download |
|---|---|---|
| `projectsidewalk/rampnet-model` (RampNet, MIT) | **Curb-ramp detection** | ~360 MB |
| `facebook/mask2former-swin-large-mapillary-vistas-semantic` | Segmentation (65 Vistas classes, run at 1024 px) | ~866 MB |
| `yolo11x.pt` (COCO) | Stop-sign boxes (`signs`) | ~115 MB |
| `IDEA-Research/grounding-dino-tiny` + `facebook/sam-vit-base` | Driveways (`driveways`) | ~689 MB + ~375 MB |

Hugging Face weights go to the HF cache (`%USERPROFILE%\.cache\huggingface`, or set `HF_HOME`).
YOLO weights go to `.cache/weights/`.

**Ramps** come from two sources, each strong where the other is weak:
- **Big, confident segmented regions.** A Vistas `Curb Cut` region counts on its own if it covers at least `RAMP_MIN_AREA_FRACTION` (0.1%) of the image with `RAMP_MIN_CONFIDENCE` (0.55). This finds driveway aprons in close-up photos, and rejects thin strips along a curb.
- **RampNet points.** [RampNet](https://huggingface.co/projectsidewalk/rampnet-model) finds far and small pedestrian ramps (and driveway aprons) are. It expects a 360° panorama, so each photo is placed at its true position in an otherwise empty panorama, using the image's estimated focal length and pitch, and the detected points are mapped back onto the photo. A point needs `RAMP_MIN_SCORE` (default 0.30) to count. It scores at most 0.19 on close-up driveway photos, where the segmenter does the work, and a point under a weaker region lets that region count too. Its weights are loaded from safetensors, so no code from the Hugging Face repository runs.

**Everything else.** `CLASS_MAP` in `config.py` maps Vistas labels to the canonical classes (for example `Pole` / `Traffic Sign Frame` / `Utility Pole` become `pole`). A ramp with no segmented region under its RampNet point is reported but not measured.
Canonical classes with no match in the model's labels are logged as warnings.

### What to detect

`--detect` (or `DETECT` in `.env`) picks the targets; the default is `all`. Each target uses the model that tested best for it, and a model is only loaded, and its weights only downloaded, if its target is chosen.

| Target | Detector | Notes |
|---|---|---|
| `ramps` | RampNet points, plus big confident segmenter regions | RampNet finds far corner ramps; the segmenter finds close-up ramps |
| `driveways` | Grounding DINO + SAM, prompt "a driveway." | each driveway that touches the road (or the photo's edge) becomes a ramp with `source` `grounding_dino+sam` and `kind: driveway`; needs a score of 0.2 |
| `signs` | YOLO | needs `poles`, which are added automatically |
| `poles`, `curbs` | the segmenter | |

```bash
python main.py detect "sussex" --detect ramps,driveways
python main.py all "sussex" --images-dir D:\photos --detect driveways
```

The segmenter always runs, because the road, sidewalk and curb it finds are what the measurement stage uses.

### Outputs

```
output/<slug>/
  masks/<image_stem>_<class>_<n>.png   binary instance masks (0/255)
  masks/<image_stem>_semantic.png      class-index map (see masks/labels.json)
  masks/labels.json                    index -> class legend and colours
  annotated/<image_stem>.jpg           overlay, stop-sign boxes, curb lines, pole bases, labels
  contact_sheet_annotated.jpg
```

Every instance in `detections.per_image[].instances` has these fields: `instance_id`, `class`, `source`, `confidence`, `bbox` `[x1,y1,x2,y2]`, `area_px`, `centroid`, and `mask`. Some classes add more:
- **stop_sign:** `bottom_center`, `traffic_sign_overlap` (the fraction of the box covered by the segmentation's traffic-sign mask), and `pole` (`instance_id` plus `base_pixel`, the pole's lowest pixel). Stop signs below `STOP_SIGN_THRESHOLD` (0.4) are dropped and logged.
- **curb_ramp:** `point` (the RampNet detection in image pixels), `confidence` (RampNet's score), and `mask_source`: `segmentation` when a ramp region was found under the point (it has a `curb_line`, and Stage 4 can measure it) or `point` when there was none (Stage 4 reports the default with that reason).
- **curb:** `curb_line`, the road pixels touching the instance, given as a left-to-right polyline.

`detections.status` is one of these values:
- `ok`
- `partial`: some images failed
- `no_detections`: no stop sign and no curb ramp
- `error`


## Stage 4 — measurement

This stage runs only on saved Stage 2 and 3 outputs and makes no Google calls. All limits, caps and tuning knobs are in `measure/config.py`.

| Measurement | Flat ground (A) | Reference scale (B) | Point cloud (C) | Triangulation |
|---|---|---|---|---|
| Stop sign height (pole base → sign bottom) | ✓ | ✓ (30 in sign face) | ✓ | ✓ (2+ panoramas, ≥ 10° apart) |
| Curb height | ✓ | – | ✓ (road/sidewalk planes near the curb) | – |
| Ramp running / cross slope | – | – | ✓ (RANSAC plane, split along/across the curb) | – |

- **A:** assumes flat ground and a camera 2.5 m above it (`CAMERA_HEIGHT_M`).
- **B:** uses the stop sign's known width as a ruler and doesn't depend on camera height.
- **C:** uses Depth Anything V2 Metric Outdoor (`DEPTH_MODEL`, ~1.34 GB; the Base/Small variants are 390/99 MB). Each depth map is rescaled to agree with flat-ground depth on nearby road, and the factor is stored per image as `depth_scale`. Depth maps are cached in `output/<slug>/depth/`.

**Fusion:** estimates are grouped per physical object by ground position, then fused.
1. Estimates outside the plausible ranges are discarded.
2. MAD outliers are rejected.
3. The median of what's left is the fused value.

**Confidence (0–1):** the product of six factors:
- number of estimates
- agreement between them
- distance from the camera
- Stage 3 mask confidence
- how close the depth-scale correction is to 1
- how many different methods contributed

Each factor is saved in `confidence_factors`.

**Measured vs default:** a value is `"source": "measured"` only when it is plausible and its confidence is at least `MEASURE_CONFIDENCE_THRESHOLD` (0.5). Otherwise the configured maximum is reported with `"source": "default_max"` and a `reason`, and any fused value is kept in `candidate_value_m`.

| Default reported when not measured | Value | Kind |
|---|---|---|
| Ramp running slope | 8.33 % (1:12) | ADA legal limit |
| Ramp cross slope | 2.08 % (1:48) | ADA legal limit |
| Stop sign height | 3.05 m (10 ft) | chosen cap; the MUTCD sets only a 7 ft minimum |
| Curb height | 0.229 m (9 in) | chosen cap; there is no federal limit |

**Flags:**
- Measured ramps get `exceeds_limit`, plus a per-slope `beyond_uncertainty` that tells a confident exceedance apart from one within noise.
- Measured signs get `below_mutcd_minimum`.

**Outputs:**
- `output/<slug>/measured/<image>.jpg`: the measured lines with their values and confidence
- `output/<slug>/pointclouds/<image>.ply`: only with `--export-pointcloud`
- Batch runs also add the measurement columns to `output/summary_<csv>.csv`.

### Evaluation

`ground_truth.csv` has the columns `address, stop_sign_height_m, curb_height_m, ramp_running_slope_pct, ramp_cross_slope_pct`, and any of them may be blank. See `ground_truth.example.csv`.

`python main.py evaluate ground_truth.csv` reports MAE, median error, max error and bias for each of these:
- each method on its own
- `fused`: measured values only
- `final`: including default fallbacks

It writes the results to `output/evaluation_<name>.json`.

## Tests

```bash
pytest              # all unit tests; HTTP and models are mocked
pytest -m slow      # stage 3 and 4 real models on a sample image (downloads weights)
```
