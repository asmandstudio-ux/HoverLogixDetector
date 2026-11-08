"""Label a folder of photos with a trained model (how the 946 'auto' labels were made).

    python training/autolabel.py --model models/hoverlogix_v1.pt --images dataset/images --out dataset/labels_auto

Rules applied after prediction (same as v1):
* per-class thresholds: pallet / pallet_label / beam / upright 0.35, location_label 0.30
* class-wise NMS at IoU 0.5, plus dropping beam/upright boxes mostly inside a stronger one
* close-up photos (a pallet_label covering > 12 % of the frame) keep only label boxes
"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

TH = {0: 0.35, 1: 0.35, 2: 0.30, 3: 0.35, 4: 0.35}


def contain(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih; A = (a[2] - a[0]) * (a[3] - a[1]); B = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (A + B - inter + 1e-9), inter / (min(A, B) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--images", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = YOLO(a.model)
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for i in range(0, len(files), 8):
        for r in m.predict([str(f) for f in files[i:i + 8]], imgsz=a.imgsz, conf=0.2, verbose=False):
            dets = []
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist()):
                if s >= TH[int(c)]:
                    x, y, w, h = b
                    dets.append([int(c), s, (x - w / 2, y - h / 2, x + w / 2, y + h / 2)])
            closeup = any(d[0] == 1 and (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]) > 0.12 for d in dets)
            dets.sort(key=lambda d: -d[1]); keep = []
            for d in dets:
                if closeup and d[0] in (0, 3, 4):
                    continue
                if any(k[0] == d[0] and (lambda q: q[0] > 0.5 or (d[0] in (3, 4) and q[1] > 0.8))(contain(k[2], d[2])) for k in keep):
                    continue
                keep.append(d)
            lines = [f"{c} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}" for c, _, (x1, y1, x2, y2) in keep]
            Path(a.out, Path(r.path).stem + ".txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"labelled {len(files)} images into {a.out}")


if __name__ == "__main__":
    main()
