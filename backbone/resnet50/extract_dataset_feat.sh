#!/usr/bin/env bash
set -e

# Sử dụng: ./extract_dataset_feat.sh --part 0 --total 1 --gpu 0
while [[ $# -gt 0 ]]; do
  case $1 in
    --part)  PART="$2"; shift;;
    --total) TOTAL="$2"; shift;;
    --gpu)   GPU="$2"; shift;;
    *)       echo "Unknown arg: $1"; exit 1;;
  esac
  shift
done

DATA_PATH="../../dataset/videos"              # thư mục .mp4
SAVE_PATH="../../dataset/features/resnet50"   # nơi lưu .npy
BATCH_SIZE=32                                 # batch size khi extract

python extract_cnn_features.py \
  --part        $PART \
  --total       $TOTAL \
  --gpu         $GPU \
  --data_path   $DATA_PATH \
  --save_path   $SAVE_PATH \
  --batch_size  $BATCH_SIZE

echo "✅ Feature extraction complete for part $PART of $TOTAL"
