"""Command line: python -m pallet_safety photo.jpg [more.jpg] --model models/hoverlogix_v1_960.onnx --out results/"""
import argparse
import json
import os

import cv2

from .pallet_lean import Detector, Limits, analyse_image, draw_result


def main():
    ap = argparse.ArgumentParser(description="Check pallets and pallet stacks for lean.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "models", "hoverlogix_v1_960.onnx"))
    ap.add_argument("--out", default="pallet_results")
    ap.add_argument("--reference", default="auto", help='"auto", "photo" or degrees')
    ap.add_argument("--pallet-mm", type=float, default=1200)
    ap.add_argument("--warn-deg", type=float, default=2.0)
    ap.add_argument("--unsafe-deg", type=float, default=4.0)
    ap.add_argument("--out-check-mm", type=float, default=100)
    ap.add_argument("--out-unsafe-mm", type=float, default=150)
    a = ap.parse_args()
    ref = a.reference if a.reference in ("auto", "photo") else float(a.reference)
    lim = Limits(a.warn_deg, a.unsafe_deg, a.out_check_mm, a.out_unsafe_mm, a.pallet_mm)
    det = Detector(a.model)
    os.makedirs(a.out, exist_ok=True)
    for path in a.images:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: could not read")
            continue
        res = analyse_image(img, det, lim, reference=ref)
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(a.out, stem + ".json"), "w") as f:
            json.dump(res, f, indent=1)
        cv2.imwrite(os.path.join(a.out, stem + "_checked.jpg"), draw_result(img, res), [cv2.IMWRITE_JPEG_QUALITY, 88])
        n_st = len(res["stacks"])
        print(f"{path}: {res['overall'].upper()}  pallets={len(res['pallets'])} stacks={n_st}")


if __name__ == "__main__":
    main()
