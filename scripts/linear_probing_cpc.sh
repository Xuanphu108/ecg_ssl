#!/bin/bash

# ==========================================
# Script: linear_probing_cpc.sh
# Description: Linear probing using a CPC pretrained model
# ==========================================

# Stop the script immediately if any command fails
set -e

# ==========================
# Default configuration
# ==========================
DATA_PATH="${DATA_PATH:-datasets/ecg_data_processed/ptb_xl_fs100/}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/downstream/ptbxl_all/cpc/linear_prob/}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-128}"
INPUT_SIZE="${INPUT_SIZE:-1000}"
FINETUNE_DATASET="${FINETUNE_DATASET:-ptbxl_all}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-runs/pretrained/cpc/all/version_0/best_model/epoch=0-step=299.ckpt}"

# ==========================
# Print configuration
# ==========================
echo "=== CPC Linear Probing Configuration ==="
echo "DATA_PATH:        $DATA_PATH"
echo "OUTPUT_PATH:      $OUTPUT_PATH"
echo "EPOCHS:           $EPOCHS"
echo "LR:               $LR"
echo "BATCH_SIZE:       $BATCH_SIZE"
echo "INPUT_SIZE:       $INPUT_SIZE"
echo "FINETUNE_DATASET: $FINETUNE_DATASET"
echo "PRETRAINED_MODEL: $PRETRAINED_MODEL"
echo "======================================"

# ===============================
# Run linear probing
# ===============================
python ./source/main_cpc.py \
  --data "$DATA_PATH" \
  --normalize 1 \
  --epochs $EPOCHS \
  --output-path "$OUTPUT_PATH" \
  --lr $LR \
  --batch-size $BATCH_SIZE \
  --input-size $INPUT_SIZE \
  --finetune \
  --pretrained "$PRETRAINED_MODEL" \
  --finetune-dataset $FINETUNE_DATASET \
  --fc-encoder \
  --train-head-only

echo "CPC linear probing completed successfully!"