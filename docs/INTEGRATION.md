# Integrating HoverLogixDetector into DroneIMS

This repository holds everything built in the "pallet detection + pallet safety" work (Nov 2026).
It is self-contained: models, dataset labels, training scripts, a Python library, an Android app and a web page.
This guide says what each part is, its exact data contract, and where it fits in DroneIMS.

## 1. What is in the repo

| Path | What | Status |
|---|---|---|
| `models/` | YOLO26n detector v1: `.pt`, 960 px `.onnx` (+ two LiteRT `.tflite` in the app assets) | working, bootstrap quality |
| `dataset/` | 984 YOLO label files, 5 classes, `label_status.csv`, previews (images live on Finney's Mac) | 36 hand, 946 auto |
| `training/` | `train.py`, `autolabel.py`, `export_models.py` | used for v1 |
| `pallet_safety/` | Python library + CLI: detection, lean check with the straight-box method, stack check | tested (`tests/`, CI) |
| `app/` + Gradle files | Android app "HoverLogix Detect": live camera + video detection, correction editor, YOLO export | builds in CI (APK under Releases) |
| `web/pallet-lean-check/` | Browser tool: upload photos, lean check, drag to correct, verdict override | live as a claude.ai artifact |
| `web/detection-review/` + `training/HoverLogix_train.ipynb` | Training loop: review/correct model labels (adds `empty_location`), export, train on Colab | v2 loop, Nov 2026 |
| `.github/workflows/` | `build-apk.yml` (APK release on app changes), `python-tests.yml` | green |

## 2. The detector

Classes: `0 pallet, 1 pallet_label, 2 location_label, 3 beam, 4 upright` (`dataset/classes.txt`).

* Input `[1,3,S,S]` float32 RGB 0..1, letterboxed: keep aspect ratio, pad with grey 114, image centred.
* Output `[1,9,N]`: rows 0..3 = cx, cy, w, h; rows 4..8 = per-class scores (already sigmoid).
  ONNX gives input-pixel coordinates; LiteRT gives normalised 0..1 (if max centre <= 2, multiply by S).
* No NMS in the graph: take the best class per anchor, threshold, then per-class NMS (IoU 0.5).
  Thresholds used everywhere: pallet 0.45, beam/upright 0.35, labels 0.30 to 0.35.
* Reference decoders in three languages: `pallet_safety/pallet_lean.py::Detector`, `app/.../Detector.kt`,
  `web/pallet-lean-check/index.html::detect`.

Quality: trained on 36 hand-labelled photos only. Good on rack scenes and label close-ups from the same
warehouse; expect misses on small location stickers and on other warehouses. Retrain before production (section 6).

## 3. Pallet safety: the straight-box lean check

Library: `pallet_safety` (numpy, opencv-python-headless, onnxruntime). Same algorithm and numbers as the web page.

```python
import cv2
from pallet_safety import Detector, Limits, analyse_image, draw_result
det = Detector("models/hoverlogix_v1_960.onnx")
res = analyse_image(cv2.imread("photo.jpg"), det, Limits(pallet_mm=1200))
cv2.imwrite("checked.jpg", draw_result(cv2.imread("photo.jpg"), res))
```

CLI: `python -m pallet_safety photos/*.jpg --out results/` writes `<name>.json` and `<name>_checked.jpg`.

How it decides:
1. Photo is scaled to a long side of 1600 px (all coordinates refer to this "work" image).
2. Per pallet: base line = straightest strong horizontal edge in the bottom 20 % of the box; load edges = strongest
   vertical edge per row near each side, RANSAC line fit.
3. Reference vertical: median tilt of rack uprights found in the photo, else the photo's vertical
   (perspective makes "perpendicular to the base" unreliable; verticals stay vertical for a level camera).
4. A straight box stands on the load's two bottom corners and rises along the reference vertical.
   `outsideMm` = how far the load top has moved sideways out of that box (scaled by `pallet_mm` / box width).
   Symmetric widening (camera angle, bulging load) is reported as `spreadMm` and not counted as lean.
5. Verdict = worse of: lean angle (CHECK >= 2 deg, UNSAFE >= 4 deg) and outside (CHECK >= 100 mm, UNSAFE >= 150 mm).
   All limits are placeholders: set them to what the customer counts as unsafe.
6. Stacks: a pallet whose bottom touches another pallet's top (>= 50 % horizontal overlap, no detected beam between)
   is stacked on it. Every level is checked against ONE box standing on the bottom pallet, so leans add up.
7. Pallets cut off by the photo edge are `not_judged`.

Result JSON (`analyse_image`):

```jsonc
{
  "workSize": [901, 1600],
  "reference": {"mode": "uprights" | "photo" | "manual", "deg": -0.54, "uprightTilts": [-0.54]},
  "limits": {"warn_deg": 2, "unsafe_deg": 4, "out_check_mm": 100, "out_unsafe_mm": 150, "pallet_mm": 1200},
  "overall": "ok" | "check" | "unsafe" | "no_pallets",
  "pallets": [{
    "id": "p4", "n": 4, "box": [x1,y1,x2,y2], "score": 0.91, "sidesFound": true, "partial": false, "call": null,
    "lines": {"left": [xTop,yTop,xBot,yBot], "right": [...], "base": [xL,yL,xR,yR]},
    "result": {"lean": 0.82, "taper": 0.8, "leftTilt": 0.1, "rightTilt": 1.5,
               "outsideMm": 22.1, "spreadMm": 14.0, "verdict": "ok" | "check" | "unsafe" | "not_judged"}
  }],
  "stacks": [{
    "pallets": ["p5", "p4"],            // bottom first
    "outsideMm": 57.0, "lean": 0.94, "verdict": "ok", "partial": false,
    "levels": [{"n": 5, "id": "p5", "lean": 0.83, "placedOffCentreMm": 0.0, "topOutsideMm": 27.0}, ...]
  }],
  "uprights": [{"box": [...], "score": 0.8}], "beams": [{"box": [...], "score": 0.7}]
}
```

`analyse_image(img, pallet_boxes=[[x1,y1,x2,y2]])` measures reviewer-drawn boxes without the model.

What it does NOT do: damage (broken boards, crushed boxes, torn wrap) and overhang beyond the pallet footprint.
Damage needs a trained class or a vision-LLM check; overhang needs a separate "pallet base" class.

## 4. Where each part fits in DroneIMS

| DroneIMS part | Use from this repo | Notes |
|---|---|---|
| Python detection / pallet-safety pipeline on the NUC | `pallet_safety` | Replace the Hough-line lean step with `analyse_image`. Keep the Claude Vision call for damage categories (crushed packaging, broken boards, rack engagement); feed it the `draw_result` image so it sees the measured boxes. |
| PostgreSQL | store `analyse_image` output | e.g. table `pallet_checks(id, photo_id, location_code, overall, result jsonb, model_version, created_at)` and `pallet_reviews(check_id, pallet_id, call, lines jsonb, reviewer, created_at)` for corrections. |
| REST API | wrap the library | `POST /pallet-checks` (image upload) -> result JSON; `PATCH /pallet-checks/{id}/pallets/{pid}` for reviewer edits (same fields the web page saves). |
| Flutter pallet inspector app | `models/*.tflite` + the box method | Load the LiteRT model with `tflite_flutter`, port the decode from `Detector.kt` (about 60 lines). For lean, either call the REST API or port `pallet_safety` (pure numeric code, no OpenCV-only features beyond Sobel/blur). It replaces the Sobel-histogram lean estimate. |
| Android MSDK app on the RC Plus 2 / drone feed | `app/.../Detector.kt` | Independent of CameraX: call `Detector.detect(bitmap, confidence)` on frames from the DJI MSDK video decoder. LiteRT runtime is bundled (no Play Services) and is arm64-only. |
| Guest scanner app (small download) | `hoverlogix_fast_640.tflite` (~10 MB float32) | Smaller with int8 quantisation (`export(..., int8=True)` + calibration images). |
| Barcode reading | pallet_label / location_label boxes | Crop each label box from the full-resolution frame, then decode (ML Kit bundled, Dynamsoft, SwiftDecoder). |

## 5. Correction data formats (feed for retraining)

**Android app export** (`hoverlogix_corrections_<date>.zip`):
```
images/<time>_<source>.jpg       full-resolution frame
labels/<time>_<source>.txt       YOLO, final boxes (rejected ones removed)
review_log.jsonl                 one line per frame:
  {"image", "source", "time", "width", "height",
   "boxes": [{"status": "confirmed|corrected|rejected|added", "class", "box": [l,t,r,b],
              "model_class", "model_box", "model_score"}]}
classes.txt, data.yaml
```

**Web page** (claude.ai artifact database, collection `photos`, one document per photo; also downloadable as JSON):
`{name, assetId, w, h, created, updated, ref{mode, manual, auto, uprightCount}, uprights[], beams[], noStack[],
stacks[{pallets, offsetMm, lean, verdict, partial}], pallets[{id, n, box, score, auto{left,right,base}, lines{...},
sidesFound, partial, call: null|"straight"|"leaning"|"not_pallet", edited, added, result{lean, taper, verdict, reference}}]}`.
Coordinates are in the stored 1600 px photo. `call` is the reviewer's ground truth for lean; use it to tune the limits.
Field names differ slightly from the Python result (`offsetMm` = `outsideMm`).

## 6. Retraining loop

1. Put photos in `dataset/images/`, review `dataset/labels/` in Roboflow / CVAT / Label Studio (YOLO + `classes.txt`).
2. Merge Android corrections (`images/` + `labels/`) into the dataset.
3. `python training/train.py --data dataset/data.yaml --model models/hoverlogix_v1.pt --device 0`
4. `python training/export_models.py --weights runs/detect/hoverlogix/weights/best.pt`
   (updates `models/` and the app assets; pushing to `main` rebuilds the APK and reruns the tests).
5. Bump the model version wherever results are stored.

Useful next classes: `pallet_base` (overhang + exact base corners), `damaged_pallet` / `damaged_load`, `empty_slot`.

## 7. Known limitations

* Detector trained on one warehouse and 36 hand labels; small location labels are the weakest class.
* Lean limits and mm limits are placeholders, not a standard.
* The camera must be roughly level (drone gimbal, or phone held upright); uprights in view correct small roll.
* Stack grouping uses boxes and beams; reviewers can split a wrong stack ("Not stacked" in the web page).
* LiteRT Android runtime is 64-bit only.
