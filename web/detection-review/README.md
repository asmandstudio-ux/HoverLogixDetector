# Warehouse Detection Review

Review desk for the training loop: upload photos, the current model (ONNX, in the browser) pre-labels them,
the reviewer approves or corrects boxes (6 classes incl. `empty_location`) and marks each pallet straight /
leaning / overhang. Approved photos export as a YOLO zip for `training/HoverLogix_train.ipynb` (Google Colab).

Live as the private claude.ai artifact "Warehouse Detection Review". It stores photos and reviews through the
claude.ai artifact runtime (`db`, `assets`, `downloads`). The model file is served next to the page as
`model-v1.wasm` (the ONNX bytes of `models/hoverlogix_v1_960.onnx`); `MODEL` in the script says which version and
how its output classes map to the 6 review classes.

Per photo document (collection `images`): `{name, key, source, assetId, w, h, created, updated,
status: new|predicted|reviewed|skipped, model, preds[{c,b,s}], uncertainty, labels[{id,c,b,from: model|suggest|user,
s, orig, lean: straight|leaning|overhang|null, leanAuto{lean,outMm,verdict,lines}}], reviewedAt, reviewedModel,
edits{kept,moved,reclassed,deleted,added,suggestAccepted,suggestRejected}}`.
