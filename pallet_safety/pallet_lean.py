"""Pallet lean check: the same logic as the Pallet Lean Check web page, as a Python library.

Pipeline for one photo
  1. Detect pallets, rack uprights and beams with the Hover Logix YOLO26n model (ONNX).
  2. For every pallet, find the base line and the left/right load edges from image gradients.
  3. Stand a straight box on the load's bottom corners, rising along the reference vertical
     (rack uprights in the photo if found, else the photo's own vertical).
  4. Measure how far the load's top corners stick out of that box (mm, scaled by the pallet width)
     and the lean angle. Judge OK / CHECK / UNSAFE against configurable limits.
  5. Group block-stacked pallets (one standing on another, no beam between them) and check every
     level against ONE box standing on the bottom pallet, so leans add up through the stack.

Only numpy, opencv-python and onnxruntime are needed. Coordinates are pixels of the working image
(long side WORK_MAX px), the same frame the web page stores.

    from pallet_safety import Detector, analyse_image, draw_result
    det = Detector("models/hoverlogix_v1_960.onnx")
    result = analyse_image(cv2.imread("photo.jpg"), det)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np

CLASS_NAMES = ["pallet", "pallet_label", "location_label", "beam", "upright"]
PALLET, PALLET_LABEL, LOCATION_LABEL, BEAM, UPRIGHT = range(5)
WORK_MAX = 1600            # photos are analysed at this long side (same as the web page)
MODEL_SIZE = 960


@dataclass
class Limits:
    warn_deg: float = 2.0          # lean angle: CHECK from
    unsafe_deg: float = 4.0        # lean angle: UNSAFE from
    out_check_mm: float = 100.0    # stock outside the box: CHECK from
    out_unsafe_mm: float = 150.0   # stock outside the box: UNSAFE from
    pallet_mm: float = 1200.0      # pallet width facing the camera (1200 Euro long side, 800 short, 1000 block)


# ----------------------------------------------------------------------------------------------
# small helpers (JS-compatible rounding so results match the web page)
# ----------------------------------------------------------------------------------------------
def _jround(v: float) -> int:
    return int(math.floor(v + 0.5))


def _rad(d): return d * math.pi / 180.0
def _deg(r): return r * 180.0 / math.pi


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def to_work_image(img_bgr: np.ndarray) -> np.ndarray:
    """Downscale so the long side is at most WORK_MAX (the frame all coordinates refer to)."""
    h, w = img_bgr.shape[:2]
    s = min(1.0, WORK_MAX / max(h, w))
    if s >= 1.0:
        return img_bgr
    return cv2.resize(img_bgr, (_jround(w * s), _jround(h * s)), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------------------------------
class Detector:
    """Runs the exported YOLO26n ONNX model (input [1,3,S,S] RGB 0..1, output [1,4+nc,N])."""

    def __init__(self, onnx_path: str, providers: Optional[list] = None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else MODEL_SIZE

    def detect(self, img_bgr: np.ndarray, classes=(PALLET, UPRIGHT, BEAM), min_score=None, nms_iou=0.5):
        """Returns [{'cls', 'name', 'score', 'box':[x1,y1,x2,y2]}] in img_bgr pixel coordinates."""
        min_score = min_score or {PALLET: 0.45, UPRIGHT: 0.35, BEAM: 0.35, PALLET_LABEL: 0.35, LOCATION_LABEL: 0.3}
        S = self.size
        H, W = img_bgr.shape[:2]
        r = min(S / W, S / H)
        nw, nh = _jround(W * r), _jround(H * r)
        px, py = (S - nw) // 2, (S - nh) // 2
        canvas = np.full((S, S, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        y = self.session.run(None, {self.input_name: x})[0][0]          # (4+nc, N)
        scores = y[4:]
        best = scores.argmax(0)
        bs = scores.max(0)
        keep_mask = bs > 0.3
        cx, cy, w, h = y[0], y[1], y[2], y[3]
        sc = S if max(cx[keep_mask].max(initial=0), cy[keep_mask].max(initial=0)) <= 2 else 1.0
        dets = []
        for a in np.nonzero(keep_mask)[0]:
            c = int(best[a])
            if c not in classes or bs[a] < min_score.get(c, 0.35):
                continue
            b = [(cx[a] - w[a] / 2) * sc, (cy[a] - h[a] / 2) * sc, (cx[a] + w[a] / 2) * sc, (cy[a] + h[a] / 2) * sc]
            b = [min(W, max(0, (b[0] - px) / r)), min(H, max(0, (b[1] - py) / r)),
                 min(W, max(0, (b[2] - px) / r)), min(H, max(0, (b[3] - py) / r))]
            dets.append({"cls": c, "name": CLASS_NAMES[c], "score": float(bs[a]), "box": [float(v) for v in b]})
        out = []
        for c in classes:
            lst = sorted([d for d in dets if d["cls"] == c], key=lambda d: -d["score"])
            while lst:
                b = lst.pop(0)
                out.append(b)
                lst = [q for q in lst if _iou(b["box"], q["box"]) <= nms_iou]
        return out


# ----------------------------------------------------------------------------------------------
# image gradients (same filters as the web page)
# ----------------------------------------------------------------------------------------------
@dataclass
class Gradients:
    W: int
    H: int
    gx: np.ndarray   # |Sobel x| of the blurred grey image
    gy: np.ndarray   # |Sobel y|


def gradients(img_bgr: np.ndarray) -> Gradients:
    rgb = img_bgr[..., ::-1].astype(np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    k = np.array([1, 4, 6, 4, 1], np.float32) / 16.0
    p = np.pad(gray, ((0, 0), (2, 2)), mode="edge")
    tmp = sum(k[j] * p[:, j:j + gray.shape[1]] for j in range(5))
    p = np.pad(tmp, ((2, 2), (0, 0)), mode="edge")
    bl = sum(k[j] * p[j:j + gray.shape[0], :] for j in range(5))
    H, W = bl.shape
    gx = np.zeros_like(bl)
    gy = np.zeros_like(bl)
    a, b, c = bl[:-2, :-2], bl[:-2, 1:-1], bl[:-2, 2:]
    d, f = bl[1:-1, :-2], bl[1:-1, 2:]
    g, h, i = bl[2:, :-2], bl[2:, 1:-1], bl[2:, 2:]
    gx[1:-1, 1:-1] = np.abs((c + 2 * f + i) - (a + 2 * d + g))
    gy[1:-1, 1:-1] = np.abs((g + 2 * h + i) - (a + 2 * b + c))
    return Gradients(W, H, gx, gy)


def _rng(seed: int):
    s = (seed & 0xFFFFFFFF) or 1
    def nxt():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 4294967296.0
    return nxt


def fit_line(ys: np.ndarray, xs: np.ndarray, tol: float):
    """Robust fit x = a*y + b (RANSAC, then least squares on inliers)."""
    n = len(ys)
    if n < 10:
        return None
    rnd = _rng(n * 7919 + _jround(float(xs[0])))
    best, best_n = None, 0
    for _ in range(300):
        i, j = int(rnd() * n), int(rnd() * n)
        if ys[i] == ys[j]:
            continue
        a = (xs[j] - xs[i]) / (ys[j] - ys[i])
        b = xs[i] - a * ys[i]
        cnt = int(np.count_nonzero(np.abs(xs - (a * ys + b)) < tol))
        if cnt > best_n:
            best_n, best = cnt, (a, b)
    if best is None:
        return None
    m = np.abs(xs - (best[0] * ys + best[1])) < tol
    yy, xx = ys[m], xs[m]
    cnt = len(yy)
    den = cnt * (yy * yy).sum() - yy.sum() ** 2
    a = (cnt * (yy * xx).sum() - yy.sum() * xx.sum()) / den if den else best[0]
    return {"a": float(a), "b": float((xx.sum() - a * yy.sum()) / cnt), "inliers": int(cnt)}


# ----------------------------------------------------------------------------------------------
# per-pallet geometry
# ----------------------------------------------------------------------------------------------
def find_base(gr: Gradients, box):
    """Top edge of the pallet base: the straightest strong horizontal edge in the bottom of the box."""
    x1, y1, x2, y2 = [_jround(v) for v in box]
    h = y2 - y1
    cx = (x1 + x2) / 2
    top = _jround(y1 + 0.80 * h)
    bot = min(gr.H - 2, _jround(y2 + 0.03 * h))
    xs = np.arange(max(1, x1), min(gr.W - 1, x2), 2)
    best = (-1.0, _jround((top + bot) / 2), 0.0)
    ang = -8.0
    while ang <= 8.001:
        t = math.tan(_rad(ang))
        offs = np.floor(t * (xs - cx) + 0.5)
        for row in range(top, bot + 1):
            yy = (row + offs).astype(int)
            ok = (yy >= 1) & (yy < gr.H - 1)
            if ok.any():
                s = float(gr.gy[yy[ok], xs[ok]].mean())
                if s > best[0]:
                    best = (s, row, ang)
        ang += 0.5
    t = math.tan(_rad(best[2]))
    return [float(x1), best[1] + t * (x1 - cx), float(x2), best[1] + t * (x2 - cx)]


def find_side(gr: Gradients, box, y_top, y_bottom, side, inward=0.15, outward=0.03):
    """Strongest vertical edge per row near one side of the box, fitted to a straight line."""
    x1, _, x2, _ = box
    w = x2 - x1
    lo = max(1, _jround(x1 - outward * w if side == "left" else x2 - inward * w))
    hi = min(gr.W - 2, _jround(x1 + inward * w if side == "left" else x2 + outward * w))
    if hi < lo:
        return None
    rows = np.arange(_jround(y_top), y_bottom - 2, 2).astype(int)
    rows = rows[(rows >= 0) & (rows < gr.H)]
    if len(rows) == 0:
        return None
    band = gr.gx[rows, lo:hi + 1]
    mx = band.max(1)
    arg = band.argmax(1) + lo
    keep = mx >= 25
    ys, xs = rows[keep].astype(float), arg[keep].astype(float)
    f = fit_line(ys, xs, max(3.0, gr.W / 400))
    if not f or f["inliers"] < 15:
        return None
    return {"line": [f["a"] * y_top + f["b"], y_top, f["a"] * y_bottom + f["b"], y_bottom], "inliers": f["inliers"]}


def analyse_pallet(gr: Gradients, box):
    """Lines for one pallet: base [xl,yl,xr,yr], left/right load edge [x_top,y_top,x_bottom,y_bottom]."""
    base = find_base(gr, box)
    y_base = min(base[1], base[3])
    y_top = box[1] + 0.03 * (box[3] - box[1])
    L = find_side(gr, box, y_top, y_base, "left")
    R = find_side(gr, box, y_top, y_base, "right")
    m = 0.008
    partial = box[0] < gr.W * m or box[2] > gr.W * (1 - m) or box[1] < gr.H * m or box[3] > gr.H * (1 - m)
    return {
        "base": base,
        "left": L["line"] if L else [box[0], y_top, box[0], y_base],
        "right": R["line"] if R else [box[2], y_top, box[2], y_base],
        "sidesFound": bool(L and R),
        "partial": bool(partial),
    }


def side_tilt(l) -> float:
    """Tilt of a load edge in degrees, + = top leans right."""
    return _deg(math.atan2(l[0] - l[2], l[3] - l[1]))


def upright_tilts(gr: Gradients, uprights):
    out = []
    for u in uprights:
        x1, y1, x2, y2 = u["box"]
        if y2 - y1 < 0.35 * gr.H:
            continue
        L = find_side(gr, u["box"], y1 + 2, y2 - 2, "left", 0.35, 0.25)
        R = find_side(gr, u["box"], y1 + 2, y2 - 2, "right", 0.35, 0.25)
        if not L or not R:
            continue
        t = (side_tilt(L["line"]) + side_tilt(R["line"])) / 2
        if abs(t) < 6:
            out.append(t)
    return out


# ----------------------------------------------------------------------------------------------
# the straight box, verdicts, stacks
# ----------------------------------------------------------------------------------------------
class Envelope:
    """Straight box standing on a load's bottom corners, rising along the reference vertical."""

    def __init__(self, base_pallet, ref_deg):
        L, R = base_pallet["lines"]["left"], base_pallet["lines"]["right"]
        self.t = math.tan(_rad(ref_deg))
        self.bl, self.br = (L[2], L[3]), (R[2], R[3])
        self.width = math.hypot(R[2] - L[2], R[3] - L[3])

    def xl(self, y): return self.bl[0] + self.t * (self.bl[1] - y)
    def xr(self, y): return self.br[0] + self.t * (self.br[1] - y)


def level_out(env: Envelope, p):
    """How far a load's top corners are outside the box (px). shift = sideways (lean), spread = both sides wider."""
    L, R = p["lines"]["left"], p["lines"]["right"]
    lo, ro = env.xl(L[1]) - L[0], R[0] - env.xr(R[1])
    lob, rob = env.xl(L[3]) - L[2], R[2] - env.xr(R[3])
    return {"leftOut": lo, "rightOut": ro, "shift": (ro - lo) / 2, "spread": (ro + lo) / 2, "placeShift": (rob - lob) / 2}


_RANK = {"ok": 0, "check": 1, "unsafe": 2}


def judge(lean_deg, shift_mm, lim: Limits) -> str:
    a = "unsafe" if abs(lean_deg) >= lim.unsafe_deg else "check" if abs(lean_deg) >= lim.warn_deg else "ok"
    m = "unsafe" if abs(shift_mm) >= lim.out_unsafe_mm else "check" if abs(shift_mm) >= lim.out_check_mm else "ok"
    return m if _RANK[m] > _RANK[a] else a


def measure(p, ref_deg, lim: Limits):
    lt, rt = side_tilt(p["lines"]["left"]), side_tilt(p["lines"]["right"])
    lean = (lt + rt) / 2 - ref_deg
    env = Envelope(p, ref_deg)
    o = level_out(env, p)
    mmpx = lim.pallet_mm / env.width if env.width > 0 else 0
    off = o["shift"] * mmpx
    return {"lean": lean, "taper": (rt - lt) / 2, "leftTilt": lt, "rightTilt": rt, "outsideMm": off,
            "spreadMm": o["spread"] * mmpx, "verdict": judge(lean, off, lim)}


def _geom(p):
    L, R = p["lines"]["left"], p["lines"]["right"]
    return {"topX": (L[0] + R[0]) / 2, "topY": min(L[1], R[1]), "botX": (L[2] + R[2]) / 2, "botY": (L[3] + R[3]) / 2}


def _h_overlap(a, b): return min(a[2], b[2]) - max(a[0], b[0])


def _beam_between(beams, lower, upper):
    ya = upper["box"][3] - 0.06 * (upper["box"][3] - upper["box"][1])
    yb = lower["box"][1] + 0.06 * (lower["box"][3] - lower["box"][1])
    for bm in beams:
        cy = (bm["box"][1] + bm["box"][3]) / 2
        if min(ya, yb) - 4 < cy < max(ya, yb) + 4 and _h_overlap(bm["box"], lower["box"]) > 0.3 * (lower["box"][2] - lower["box"][0]):
            return True
    return False


def find_stacks(pallets, beams, not_stacked_ids=()):
    """Chains of pallets standing on each other (bottom first). A beam between two pallets = separate rack levels."""
    ps = [p for p in pallets if p.get("call") != "not_pallet"]
    no = set(not_stacked_ids)
    below = {}
    for U in ps:
        if U["id"] in no:
            continue
        best, best_gap = None, float("inf")
        for L in ps:
            if L is U:
                continue
            lw, uw, lh = L["box"][2] - L["box"][0], U["box"][2] - U["box"][0], L["box"][3] - L["box"][1]
            if _h_overlap(U["box"], L["box"]) < 0.5 * min(lw, uw):
                continue
            if (U["box"][1] + U["box"][3]) / 2 >= (L["box"][1] + L["box"][3]) / 2:
                continue
            gap = abs(U["box"][3] - L["box"][1])
            if gap > 0.12 * lh or _beam_between(beams, L, U):
                continue
            if gap < best_gap:
                best, best_gap = L, gap
        if best is not None:
            below[U["id"]] = best
    lowers = {p["id"] for p in below.values()}
    stacks = []
    for B in ps:
        if B["id"] not in lowers or B["id"] in below:
            continue
        chain, cur = [B], B
        while True:
            up = next((q for q in ps if below.get(q["id"]) is cur and q not in chain), None)
            if up is None:
                break
            chain.append(up)
            cur = up
        if len(chain) > 1:
            stacks.append(chain)
    return stacks


def measure_stack(chain, ref_deg, lim: Limits):
    """Every level against ONE box standing on the bottom pallet."""
    env = Envelope(chain[0], ref_deg)
    mmpx = lim.pallet_mm / env.width if env.width > 0 else 0
    bot, top = _geom(chain[0]), _geom(chain[-1])
    height = bot["botY"] - top["topY"]
    levels = []
    for p in chain:
        o = level_out(env, p)
        levels.append({"n": p["n"], "id": p["id"], "lean": measure(p, ref_deg, lim)["lean"],
                       "placedOffCentreMm": o["placeShift"] * mmpx, "topOutsideMm": o["shift"] * mmpx})
    worst = max(levels, key=lambda l: abs(l["topOutsideMm"]))
    lean = _deg(math.atan2(levels[-1]["topOutsideMm"] / (mmpx or 1), height))
    return {"pallets": [p["id"] for p in chain], "levels": levels, "outsideMm": worst["topOutsideMm"], "lean": lean,
            "verdict": judge(lean, worst["topOutsideMm"], lim), "partial": any(p["partial"] and not p.get("call") for p in chain)}


# ----------------------------------------------------------------------------------------------
# whole photo
# ----------------------------------------------------------------------------------------------
def analyse_image(img_bgr: np.ndarray, detector: Optional[Detector] = None, limits: Optional[Limits] = None,
                  reference="auto", detections=None, pallet_boxes=None):
    """Analyse one photo.

    reference: "auto" (rack uprights, else photo vertical), "photo" (0 deg) or a number in degrees.
    detections: optional precomputed detections (same format as Detector.detect) instead of running the model.
    pallet_boxes: optional list of [x1,y1,x2,y2] pallet boxes (e.g. drawn by a reviewer), skips the model.
    Coordinates in the result refer to the WORK image (long side <= 1600 px); 'workSize' gives its size.
    """
    lim = limits or Limits()
    img = to_work_image(img_bgr)
    H, W = img.shape[:2]
    if pallet_boxes is not None:
        dets = [{"cls": PALLET, "name": "pallet", "score": 1.0, "box": list(map(float, b))} for b in pallet_boxes]
    elif detections is not None:
        dets = detections
    else:
        if detector is None:
            raise ValueError("pass a Detector, detections or pallet_boxes")
        dets = detector.detect(img)
    gr = gradients(img)
    uprights = [d for d in dets if d["cls"] == UPRIGHT]
    beams = [d for d in dets if d["cls"] == BEAM]
    tilts = upright_tilts(gr, uprights)
    if reference == "auto":
        ref = float(np.median(tilts)) if tilts else 0.0
        ref_src = "uprights" if tilts else "photo"
    elif reference == "photo":
        ref, ref_src = 0.0, "photo"
    else:
        ref, ref_src = float(reference), "manual"

    pallets = []
    for i, d in enumerate(sorted([d for d in dets if d["cls"] == PALLET], key=lambda d: d["box"][0])):
        a = analyse_pallet(gr, d["box"])
        p = {"id": f"p{i + 1}", "n": i + 1, "box": [round(v) for v in d["box"]], "score": round(d["score"], 3),
             "lines": {"left": a["left"], "right": a["right"], "base": a["base"]},
             "sidesFound": a["sidesFound"], "partial": a["partial"], "call": None}
        m = measure(p, ref, lim)
        p["result"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()}
        if p["partial"]:
            p["result"]["verdict"] = "not_judged"
        pallets.append(p)

    stacks = [measure_stack(ch, ref, lim) for ch in find_stacks(pallets, beams)]
    for s in stacks:
        s["outsideMm"] = round(s["outsideMm"], 1)
        s["lean"] = round(s["lean"], 2)
        for l in s["levels"]:
            for k in ("lean", "placedOffCentreMm", "topOutsideMm"):
                l[k] = round(l[k], 2)
    judged = [p["result"]["verdict"] for p in pallets if p["result"]["verdict"] != "not_judged"]
    judged += [s["verdict"] for s in stacks if not s["partial"]]
    overall = max(judged, key=lambda v: _RANK[v]) if judged else "no_pallets"
    return {
        "workSize": [W, H], "reference": {"mode": ref_src, "deg": round(ref, 2), "uprightTilts": [round(t, 2) for t in tilts]},
        "limits": asdict(lim), "pallets": pallets, "stacks": stacks, "overall": overall,
        "uprights": [{"box": [round(v) for v in u["box"]], "score": round(u["score"], 3)} for u in uprights],
        "beams": [{"box": [round(v) for v in b["box"]], "score": round(b["score"], 3)} for b in beams],
    }


# ----------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------
_COL = {"ok": (122, 210, 47), "check": (61, 197, 255), "unsafe": (71, 82, 255), "not_judged": (207, 196, 184)}  # BGR


def draw_result(img_bgr: np.ndarray, result: dict) -> np.ndarray:
    """Annotated work-size image: straight boxes, load edges, red where stock is outside the box."""
    img = to_work_image(img_bgr).copy()
    ref = result["reference"]["deg"]
    by_id = {p["id"]: p for p in result["pallets"]}
    stacked = {pid for s in result["stacks"] for pid in s["pallets"]}
    over = img.copy()

    def box_poly(env, top_y):
        return np.array([env.bl, env.br, (env.xr(top_y), top_y), (env.xl(top_y), top_y)], np.int32)

    def shade(env, p):
        L, R = p["lines"]["left"], p["lines"]["right"]
        if env.xl(L[1]) - L[0] > 1:
            cv2.fillPoly(over, [np.array([(L[2], L[3]), (L[0], L[1]), (env.xl(L[1]), L[1]), (env.xl(L[3]), L[3])], np.int32)], (60, 70, 255))
        if R[0] - env.xr(R[1]) > 1:
            cv2.fillPoly(over, [np.array([(R[2], R[3]), (R[0], R[1]), (env.xr(R[1]), R[1]), (env.xr(R[3]), R[3])], np.int32)], (60, 70, 255))

    jobs = []
    for p in result["pallets"]:
        if p["id"] not in stacked:
            env = Envelope(p, ref)
            shade(env, p)
            jobs.append((env, min(p["lines"]["left"][1], p["lines"]["right"][1]), p["result"]["verdict"],
                         p["box"], f'{p["result"]["verdict"].upper()} {abs(p["result"]["outsideMm"]):.0f}mm'))
    for s in result["stacks"]:
        chain = [by_id[i] for i in s["pallets"]]
        env = Envelope(chain[0], ref)
        for p in chain:
            shade(env, p)
        v = "not_judged" if s["partial"] else s["verdict"]
        jobs.append((env, _geom(chain[-1])["topY"], v, chain[-1]["box"], f'STACK x{len(chain)} {v.upper()} {abs(s["outsideMm"]):.0f}mm'))
    img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    for env, top_y, verdict, box, text in jobs:
        cv2.polylines(img, [box_poly(env, top_y)], True, _COL[verdict], 3, cv2.LINE_AA)
        cv2.line(img, tuple(map(int, env.bl)), tuple(map(int, env.br)), (26, 159, 255), 4, cv2.LINE_AA)
        x, y = int(box[0]), max(18, int(box[1]) - 8)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COL[verdict], 2, cv2.LINE_AA)
    for p in result["pallets"]:
        for l in (p["lines"]["left"], p["lines"]["right"]):
            cv2.line(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 255, 255), 2, cv2.LINE_AA)
    return img
