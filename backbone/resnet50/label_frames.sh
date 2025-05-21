#!/usr/bin/env bash
set -euo pipefail

# ── CONFIG ─────────────────────────────────────────────────────────
FEAT_DIR="../../dataset/features/resnet50"           # nơi chứa *_times.npy và *_feats.npy
ANN_CSV="../../dataset/annotations/APTOS_train-val_annotation.csv"     # file CSV annotation
OUT_DIR="../../dataset/labels"       # nơi lưu *_labels.npy (mặc định FEAT_DIR)
# ─────────────────────────────────────────────────────────────────

mkdir -p "$OUT_DIR"

echo "🏷️  Generating labels:"
python3 extract_labels.py \
  --feat_dir "$FEAT_DIR" \
  --ann_csv  "$ANN_CSV" \
  --out_dir  "$OUT_DIR"

echo "✅ All done! Labels saved to $OUT_DIR"
