#!/bin/bash

# Create directories if they don't exist
mkdir -p models
mkdir -p logs

# Training parameters
FEAT_DIR="../dataset/features/resnet50"  # Directory containing *_feats.npy files
LABEL_DIR="../dataset/labels"   # Directory containing *_labels.npy files
OUT_MODEL="./models/mstcn_phase.pth"
EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=1e-3
SEQ_LEN=512
STRIDE=256

# Check if data directories exist
if [ ! -d "$FEAT_DIR" ]; then
    echo "Error: Features directory $FEAT_DIR does not exist"
    exit 1
fi

if [ ! -d "$LABEL_DIR" ]; then
    echo "Error: Labels directory $LABEL_DIR does not exist"
    exit 1
fi

# Check if there are any feature files
if [ -z "$(ls -A $FEAT_DIR/*_feats.npy 2>/dev/null)" ]; then
    echo "Error: No *_feats.npy files found in $FEAT_DIR"
    exit 1
fi

# Check if there are any label files
if [ -z "$(ls -A $LABEL_DIR/*_labels.npy 2>/dev/null)" ]; then
    echo "Error: No *_labels.npy files found in $LABEL_DIR"
    exit 1
fi

# Training command
echo "Starting training..."
python train_mstcn.py \
    --feat_dir $FEAT_DIR \
    --label_dir $LABEL_DIR \
    --out_model $OUT_MODEL \
    --epochs $EPOCHS \
    --bs $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --seq_len $SEQ_LEN \
    --stride $STRIDE \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# Prediction example (uncomment and modify as needed)
# PREDICT_FEAT="path/to/test_feats.npy"
# PREDICT_LABEL="path/to/test_labels.npy"
# 
# echo "Running prediction..."
# python train_mstcn.py \
#     --predict $PREDICT_FEAT \
#     --label_file $PREDICT_LABEL \
#     --out_model $OUT_MODEL \
#     2>&1 | tee logs/prediction_$(date +%Y%m%d_%H%M%S).log 