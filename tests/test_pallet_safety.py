"""Regression test: the two example photos must keep giving the same verdicts as the web page."""
from pathlib import Path

import cv2

from pallet_safety import Detector, analyse_image

ROOT = Path(__file__).resolve().parents[1]
DET = Detector(str(ROOT / "models/hoverlogix_v1_960.onnx"))


def _main_stack(res):
    full = [s for s in res["stacks"] if not s["partial"]]
    assert len(full) == 1, full
    return full[0]


def test_leaning_stack_is_unsafe():
    res = analyse_image(cv2.imread(str(ROOT / "tests/leaning.jpg")), DET)
    st = _main_stack(res)
    assert st["verdict"] == "unsafe"
    assert -360 < st["outsideMm"] < -270          # web page: 315 mm out to the left
    assert res["overall"] == "unsafe"


def test_straight_stack_is_ok():
    res = analyse_image(cv2.imread(str(ROOT / "tests/straight.jpg")), DET)
    st = _main_stack(res)
    assert st["verdict"] == "ok"
    assert abs(st["outsideMm"]) < 100
    assert res["reference"]["mode"] == "uprights"


def test_hand_drawn_box_without_model():
    img = cv2.imread(str(ROOT / "tests/straight.jpg"))
    res = analyse_image(img, pallet_boxes=[[202, 830, 544, 1277]])
    assert len(res["pallets"]) == 1 and res["pallets"][0]["result"]["verdict"] == "ok"
