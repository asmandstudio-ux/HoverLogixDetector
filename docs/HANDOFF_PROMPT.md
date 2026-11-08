Paste this into the DroneIMS coding chat:

---

I want to integrate the work in my private GitHub repo **asmandstudio-ux/HoverLogixDetector** into the DroneIMS codebase.
Please attach that repo to this session and read `docs/INTEGRATION.md` first, it explains every part and its data format.

What's in it:
- `models/`: my YOLO26n detector v1 (classes pallet, pallet_label, location_label, beam, upright) as .pt, 960 px ONNX, and 640/960 px LiteRT in `app/src/main/assets/`.
- `pallet_safety/`: Python library + CLI for the pallet lean check (straight box standing on each pallet's base, stock outside the box measured in mm, block-stacked pallets checked against one box on the bottom pallet). Tested in `tests/`.
- `app/`: Android app with live camera + video detection and a correction editor that exports YOLO training data.
- `web/pallet-lean-check/`: browser version of the lean check with drag-to-correct.
- `dataset/` + `training/`: labels for 984 photos and the train / auto-label / export scripts.

Please:
1. Replace the Hough-line lean step in our Python pallet-safety pipeline with `pallet_safety.analyse_image`, and keep the Claude Vision call for damage checks.
2. Store results in PostgreSQL (suggested tables are in section 4 of INTEGRATION.md) and expose them through our REST API, including an endpoint for reviewer corrections.
3. Use the LiteRT model and the box method in the Flutter pallet inspector app in place of the Sobel-histogram lean estimate.
4. Tell me what else should come across (Android correction editor, drone-feed detection via `Detector.kt`) and in what order.
Check with me before changing anything that affects the NUC deployment.
