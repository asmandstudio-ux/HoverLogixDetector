#!/usr/bin/env bash
# Assemble a static, self-hostable copy of the Pallet Lean Check page in ./dist
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$here/../.."
out="$here/dist"; rm -rf "$out"; mkdir -p "$out/examples"
cp "$here/index.html" "$out/body.html"
cp "$here"/examples/*.jpg "$out/examples/"
cp "$root/models/hoverlogix_v1_960.onnx" "$out/model-onnx.wasm"     # the page fetches the ONNX bytes under this name
tmp="$(mktemp -d)"; (cd "$tmp" && npm pack onnxruntime-web@1.20.1 >/dev/null && tar xzf onnxruntime-web-1.20.1.tgz)
cp "$tmp/package/dist/ort-wasm-simd-threaded.wasm" "$tmp/package/dist/ort-wasm-simd-threaded.mjs" "$out/"
rm -rf "$tmp"
# index.html is written as a page body (claude.ai adds the document skeleton); wrap it for normal hosting
{ printf '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>'
  cat "$out/body.html"; printf '</body></html>'; } > "$out/index.html"; rm "$out/body.html"
echo "Built $out. Serve it over http(s), e.g.: python3 -m http.server -d $out 8080"
