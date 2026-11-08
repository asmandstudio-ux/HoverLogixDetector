# Warehouse detection dataset (v1)

984 warehouse photos, YOLO labels for 5 classes:

| id | class | what it is |
|---|---|---|
| 0 | pallet | the whole load unit (pallet base + goods) |
| 1 | pallet_label | white SSCC / shipping sheet on a load |
| 2 | location_label | rack location sticker on a beam |
| 3 | beam | horizontal rack beam |
| 4 | upright | vertical rack upright |

* `labels/<photo>.txt`: `class cx cy w h`, normalised 0..1.
* `label_status.csv`: per photo `hand` (36 drawn by hand), `auto` (946 from the v1 model, not individually reviewed) or `needs_manual` (2 with no boxes), plus box counts per class.
* `previews/`: random samples with boxes drawn.

**Images are not in the repo** (1.2 GB). They are on Finney's Mac in
`Documents/Hover Logix/warehouse-images/all_images/`. File names start with the rack location they were
taken at (e.g. `K_19_1_B__<id>.jpg`), and `manifest.csv` in that folder maps each file to its original folder.
Copy them into `dataset/images/` to train.

Close-up label photos only carry `pallet_label` boxes (the load fills the frame, so no `pallet` box).
Auto labels should be reviewed (Roboflow / CVAT / Label Studio import YOLO + `classes.txt`) before a final training run.
Corrections exported from the Android app (`hoverlogix_corrections/images` + `labels`) use the same format and can be merged in directly.
