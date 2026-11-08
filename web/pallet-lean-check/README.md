# Pallet Lean Check (web page)

Single-page tool: upload warehouse photos, the YOLO model runs in the browser (onnxruntime-web, WASM),
every pallet gets a straight box standing on its base, stock outside the box is shaded red and measured in mm.
Block-stacked pallets are checked against one box on the bottom pallet. Reviewers can drag the load corners,
override the verdict (Straight / Leaning / Not a pallet), draw missed pallets, and tune the limits.

Live version: the private claude.ai artifact "Pallet Lean Check" (Finney's account). There it saves photos,
corrections and settings through the claude.ai artifact runtime (`window.claude.use("db" | "assets" | "downloads")`).
Outside claude.ai those calls are absent; the page then still works but says corrections are not saved.

## Run it yourself

```bash
./build.sh                                  # needs npm; fetches onnxruntime-web 1.20.1 runtime files
python3 -m http.server -d dist 8080         # open http://localhost:8080
```

## Integrating into DroneIMS

* The measurement logic is identical to `pallet_safety/pallet_lean.py` (same filters, same RANSAC seed),
  so the backend and the page give the same numbers. Server-side checks should call the Python module.
* To save reviews in DroneIMS instead of claude.ai, replace `initStorage()`, `savePhoto()` and the
  `state.assets.upload` call with requests to your API. The document saved per photo is built by `serialize(photo)`;
  its shape is documented in `docs/INTEGRATION.md`.
