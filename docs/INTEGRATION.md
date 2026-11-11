# Integrating HoverLogixDetector into DroneIMS

This repository holds everything built in the "pallet detection + pallet safety" work (Nov 2026).
It is self-contained: models, dataset labels, training scripts, a Python library, an Android app and a web page.
This guide says what each part is, its exact data contract, and where it fits in DroneIMS.

## 1. What is in the repo

| Path | What | Status |
|---|---|---|
| `models/` | v1: YOLO26n, 5 classes (`.pt`, 960 px `.onnx`). `models/v2/`: YOLOv8n, 7 classes (`.pt`, 640 px `.onnx`, test results); its LiteRT files are in the app assets | v2 current |
| `dataset/` | 984 YOLO label files, 5 classes, `label_status.csv`, previews (images live on Finney's Mac) | 36 hand, 946 auto |
| `training/` | `train.py`, `autolabel.py`, `export_models.py` | used for v1 |
| `pallet_safety/` | Python library + CLI: detection, lean check with the straight-box method, stack check | tested (`tests/`, CI) |
| `app/` + Gradle files | Android app "HoverLogix Detect": live camera + video detection, correction editor, YOLO export | builds in CI (APK under Releases) |
| `web/pallet-lean-check/` | Browser tool: upload photos, lean check, drag to correct, verdict override | live as a claude.ai artifact |
| `web/detection-review/` + `training/HoverLogix_train.ipynb` | Training loop: review/correct model labels (adds `empty_location`), export, train on Colab | v2 loop, Nov 2026 |
| `.github/workflows/` | `build-apk.yml` (APK release on app changes), `python-tests.yml` | green |

## 2. The detector

**v2 (current, Nov 2026): YOLOv8n, 640 px, 7 classes** `0 pallet, 1 pallet_label, 2 location_label, 3 beam, 4 upright, 5 empty_location, 6 pallet_base`.
Trained on Colab T4 with `training/HoverLogix_YOLOv8_v2.ipynb` (two stages: train, fill in classes each source never labelled, retrain).
Data: 984 DroneIMS photos + Roboflow "pallet detection" v5/v6 (pallet bases, CC BY 4.0) + "Warehouse Object Detection" (CC BY 4.0); build scripts in `training/v2/`.

Held-out test (36 hand-labelled DroneIMS photos; 177 public pallet photos), mAP50:

| Model file | DroneIMS all | pallet | pallet_label | location_label | beam | upright | pallet_base (public) | CPU speed* |
|---|---|---|---|---|---|---|---|---|
| `hoverlogix_v2.pt` | 0.85 | 0.96 | 0.97 | 0.79 | 0.68 | 0.86 | 0.84 | |
| `hoverlogix_v2_640_float32.tflite` (app "Accurate", default) | 0.82 | 0.97 | 0.97 | 0.76 | 0.64 | 0.75 | 0.82 | 1x |
| `hoverlogix_v2_640_int8.tflite` (app "Fast") | 0.67 | 0.90 | 0.84 | 0.46 | 0.51 | 0.66 | 0.79 | ~4x faster |

*Relative, measured with LiteRT on x86 CPU. TFLite input `[1,3,640,640]` float32 RGB 0..1 (NCHW), output `[1,11,8400]` normalised.
Known gaps: `empty_location` has too few examples to be reliable (rack-gap heuristic in the review page still applies); `pallet_base` is learned from
public photos and is not yet found under loaded pallets on DroneIMS photos. Approving photos in the review page fixes both in the next round.

v1 (Nov 2026) classes: `0 pallet, 1 pallet_label, 2 location_label, 3 beam, 4 upright` (`dataset/classes.txt`).

* Input `[1,3,S,S]` float32 RGB 0..1, letterboxed: keep aspect ratio, pad with grey 114, image centred.
* Output `[1,4+nc,N]` (v1: `[1,9,N]`, v2: `[1,11,8400]`): rows 0..3 = cx, cy, w, h; then per-class scores (already sigmoid).
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

## 8. Guided pallet scan (app "Scan pallets", Nov 2026)

Flow in `ScanActivity.kt`: live detection until a `pallet_label` and a `location_label` are both in view, then
1. match location labels to the pallet (`ScanLogic.locationCandidates`): labels within the pallet's width on the beam
   it stands on or the beam above it; stacked pallets count as one column; up to 4 candidates, matching ones first;
2. zoom + focus on those location labels, take a full-resolution still, crop and read each (barcode, then text);
3. zoom + focus on the pallet label, take a still, crop it, read barcodes and text;
4. the user confirms or picks the location (or types it) and saves.

Zoom rule (`ScanLogic.zoomFor`): a phone zooms around the frame centre, so the zoom is the largest that keeps the target
in the picture (`0.44 / furthest edge from centre`), capped at 8x and at the lens maximum. After zooming, boxes are found
again in the still (`relocate`) because hands move.

Telephoto and aiming (build 8): each label gets a wanted total zoom so it fills a set share of the photo
(`PALLET_LABEL_FILL` 0.4, `LOCATION_LABEL_FILL` 0.25, capped at `MAX_ZOOM` = 8 on phones). `findLenses` lists the back
cameras with their magnification relative to the main lens; `planLens` picks the strongest lens not above the wanted zoom
and the rest is digital. If the label can reach that zoom on the main lens while staying in the picture, the scan is
automatic. Otherwise the app switches lens, shows an aiming frame, and fires a burst of 4 photos once the label is centred
and steady (`aimAndBurst`); the sharpest photo that decodes is used. For the drone, use 4x to 6x optical zoom with the
gimbal centring the label, so no aiming step is needed.

Fast flow (build 9, phones without a reachable telephoto lens, e.g. S24 Ultra where Samsung exposes only 1.0x and 0.6x):
`runScanFast` takes one photo without zoom and reads the location labels from it on a second thread while the camera
zooms to the pallet label; it stops at the first pallet photo whose barcodes decode (at most 3) and zooms to the location
labels only when the first photo gave no location ID (`LabelRead.locationId`: barcode, else text like "AT 03 6C").
Every step is timed into `capture.timing_ms` in `scan.json`. Target: 3 to 5 s per pallet.

Reading (`LabelReader`): ML Kit bundled barcode model, ZXing as fallback (pure Java, no Google services), ML Kit Latin OCR.
Saved per scan (`ScanStore`, export from the home screen as `hoverlogix_scans_<date>.zip`):

```
scans/<time>/pallet_label.jpg     zoomed full-resolution crop: send this to server OCR / barcode decoding
scans/<time>/location_<n>.jpg     candidate location labels
scans/<time>/scan.json            {time, device, location_id, location_selected_candidate, location_zoom,
                                   location_candidates:[{image, matched_to_pallet, barcodes:[{format,value}], text, reader}],
                                   pallet_label:{image, width, height, zoom, barcodes, text, reader}}
scans/scans.jsonl                 one line per scan
```

DJI RC Plus 2 / drone feed: the controller has no rear camera, so there the frames come from the drone through MSDK.
Only two functions are device-specific: `zoomTo(target)` (drone: camera zoom ratio + gimbal to centre the target, which
removes the off-centre zoom limit) and `grabStill()` (drone: high-resolution photo or a decoded video frame). Everything
else (`ScanLogic`, `LabelReader`, `ScanStore`, the result screen) is reused as is.

## 7. Known limitations

* Detector trained on one warehouse and 36 hand labels; small location labels are the weakest class.
* Lean limits and mm limits are placeholders, not a standard.
* The camera must be roughly level (drone gimbal, or phone held upright); uprights in view correct small roll.
* Stack grouping uses boxes and beams; reviewers can split a wrong stack ("Not stacked" in the web page).
* LiteRT Android runtime is 64-bit only.
