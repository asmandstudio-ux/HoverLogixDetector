"""Export a trained .pt to every format the Hover Logix software uses.

    pip install ultralytics onnx onnxslim litert-torch ai-edge-litert
    python training/export_models.py --weights runs/detect/hoverlogix/weights/best.pt

Writes
  models/hoverlogix_v1_960.onnx                               backend (pallet_safety) + web page
  app/src/main/assets/hoverlogix_fast_640.tflite              Android live view
  app/src/main/assets/hoverlogix_accurate_960.tflite          Android video + editor
All three have input [1,3,S,S] RGB 0..1 (letterboxed, grey 114) and output [1,9,N]
(cx, cy, w, h + 5 class scores; per-class NMS is done by the caller).
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
a = ap.parse_args()
root = Path(__file__).resolve().parents[1]
m = YOLO(a.weights)
shutil.copy(m.export(format="onnx", imgsz=960, opset=17, simplify=True), root / "models/hoverlogix_v1_960.onnx")
shutil.copy(m.export(format="litert", imgsz=640), root / "app/src/main/assets/hoverlogix_fast_640.tflite")
shutil.copy(m.export(format="litert", imgsz=960), root / "app/src/main/assets/hoverlogix_accurate_960.tflite")
print("done: models/ and app/src/main/assets/ updated")
