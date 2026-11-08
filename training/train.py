"""Train the Hover Logix detector.

    pip install ultralytics
    python training/train.py --data dataset/data.yaml                       # from COCO-pretrained yolo26n
    python training/train.py --data dataset/data.yaml --model models/hoverlogix_v1.pt   # fine-tune v1

v1 was trained on 36 hand-labelled seed images (20 rack scenes, 16 label close-ups), CPU only:
imgsz 960, 80 epochs, batch 4, mosaic 1.0, scale 0.4, degrees 3, fliplr 0.5.
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset/data.yaml")
ap.add_argument("--model", default="yolo26n.pt")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--device", default=None, help="0 for GPU, cpu, mps on Apple silicon")
ap.add_argument("--name", default="hoverlogix")
a = ap.parse_args()

YOLO(a.model).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, name=a.name,
                    degrees=3, scale=0.4, mosaic=1.0, close_mosaic=10, fliplr=0.5)
