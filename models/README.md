# Hover Logix detector v1 (YOLO26n)

| file | use |
|---|---|
| `hoverlogix_v1.pt` | PyTorch weights (Ultralytics). Fine-tune or re-export from this. |
| `hoverlogix_v1_960.onnx` | 960 px ONNX. Used by `pallet_safety/` (backend) and the web page. |
| `../app/src/main/assets/hoverlogix_fast_640.tflite` | 640 px LiteRT, Android live view |
| `../app/src/main/assets/hoverlogix_accurate_960.tflite` | 960 px LiteRT, Android video and editor |

Classes: `0 pallet, 1 pallet_label, 2 location_label, 3 beam, 4 upright`.

**Input** `[1,3,S,S]` float RGB 0..1, letterboxed (keep aspect, pad grey 114, centred).
**Output** `[1,9,N]`: rows 0..3 = cx, cy, w, h; rows 4..8 = class scores (already sigmoid).
Boxes come out in input pixels from ONNX and normalised 0..1 from LiteRT: if the largest centre is <= 2, multiply by S.
No NMS inside the model: apply per-class NMS (IoU 0.5 to 0.6) yourself.
Reference decoders: `pallet_safety/pallet_lean.py` (Python), `app/.../Detector.kt` (Kotlin), `web/pallet-lean-check/index.html` (JS).

**Quality**: trained on only 36 hand-labelled images, so treat it as a bootstrap model.
Against the full .pt at 960 px, the 960 px LiteRT model finds ~89 % of the same boxes, the 640 px one ~77 %
(most losses are small location labels). Retrain on reviewed labels plus app corrections before relying on it.
