# HoverLogix Detect (Android)

On-device detection of **pallets, pallet labels, location labels, beams and uprights**
from the live camera or a recorded video, with a correction editor that saves
reviewed frames as a YOLO training dataset.

## Download and install (no Android Studio needed)

Every push to `main` is built automatically by GitHub Actions (*Actions* tab).
The installable APK is attached to the newest entry under **Releases**.

1. On the phone or controller, open the repository's Releases page and download `HoverLogixDetect-buildN.apk`
   (or download it on a computer and copy it over).
2. Open the file and allow "Install unknown apps" for the browser or file manager when Android asks.
3. New builds install over the old one; saved corrections are kept.

## Build it yourself in Android Studio

1. Install Android Studio (Koala or newer) and open this folder (`HoverLogixDetect`).
2. Let Gradle sync (it downloads the Android plugin, CameraX and LiteRT).
3. Connect the phone or the DJI RC Plus 2 with USB debugging on and press **Run**.
   To get an installable file instead: *Build > Build App Bundle(s) / APK(s) > Build APK(s)*.

Requirements: Android 8.0 (API 26) or newer. No Google Play Services needed.

## Screens

**Live camera**: detections drawn over the camera feed, with counts per class, fps and inference time.
- *Fast 640 / Accurate 960* toggles the model (Fast is smoother, Accurate finds more small location labels).
- *Min. confidence* slider hides weak detections.
- *Freeze & correct* takes a full-resolution photo and opens it in the editor.

**Recorded video**: pick any video, play it with detections drawn on top.
- *◀ Frame / Frame ▶* step frame by frame, the seek bar scrubs.
- *Correct frame* grabs the full-resolution frame at that moment and opens the editor.

**Editor** (the frame is re-detected with the Accurate model):
- Tap a box to select it (the smallest box under your finger wins, so labels inside pallets are easy to pick).
- **Mark wrong**: flags a false detection (shown red with a cross; tap *Restore* to undo).
- **Fix the boundary**: drag inside the box to move it, drag a corner handle to resize.
- **Wrong class**: tap one of the coloured class buttons while a box is selected.
- **Missed object**: tap the class, then *+ Add box* and drag on the image.
- Pinch to zoom, two fingers to pan. *Undo* and *Reset* (back to the model's box) are available.
- **Save** writes the frame and the corrected boxes.

## Corrections dataset

Saved on the device under `Android/data/nl.hoverlogix.detect/files/corrections/`:

```
images/<time>_<source>.jpg
labels/<time>_<source>.txt     YOLO format: class cx cy w h (normalised), rejected boxes excluded
review_log.jsonl               per box: confirmed / corrected / rejected / added, model box + score vs final box
classes.txt, data.yaml
```

*Export corrections* on the start screen zips this folder and opens the share sheet
(e-mail, Drive, Nearby Share, etc.).

## Retraining with the corrections

Copy `images/` and `labels/` from the export into your training set, then:

```bash
yolo detect train data=data.yaml model=yolo26n.pt imgsz=960 epochs=100
yolo export model=runs/detect/train/weights/best.pt format=litert imgsz=640   # -> hoverlogix_fast_640.tflite
yolo export model=runs/detect/train/weights/best.pt format=litert imgsz=960   # -> hoverlogix_accurate_960.tflite
```

Replace the two files in `app/src/main/assets/` (same names) and rebuild.
`review_log.jsonl` tells you which classes the model gets wrong most often.

## Model files

| File | Input | Notes |
|---|---|---|
| `hoverlogix_fast_640.tflite` | 640 x 640 | live view |
| `hoverlogix_accurate_960.tflite` | 960 x 960 | video review and the editor |

Both take `[1,3,S,S]` float RGB 0..1 (letterboxed, grey 114) and output `[1,9,N]`
(cx, cy, w, h normalised + 5 class scores). `Detector.kt` does letterboxing, decoding and per-class NMS.

## Code map

- `Detector.kt`: LiteRT inference, letterbox, decode, NMS
- `LiveActivity.kt`: CameraX preview + analysis + full-res capture
- `VideoActivity.kt`: MediaPlayer on a TextureView, frame stepping, full-res frame grab
- `EditorActivity.kt`, `BoxEditorView.kt`: review and correction UI
- `CorrectionStore.kt`: YOLO dataset writer and zip export

## Drone camera feed

The live screen uses the device camera. To run detection on the drone's video inside DroneIMS,
feed frames from the DJI Mobile SDK into `Detector.detect(bitmap, confidence)`: it is
independent of CameraX and can be reused as is.
