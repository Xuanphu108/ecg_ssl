#!/bin/bash

# ==========================================
# Script: pretraining_masked_transformer.sh
# Description: Run masked pretraining on multiple ECG datasets
# ===============================

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================
# Default configuration
# ==========================
DATA1="${DATA1:-datasets/ecg_data_processed/cinc_fs100_full/}"
DATA2="${DATA2:-datasets/ecg_data_processed/zheng_fs100/}"
DATA3="${DATA3:-datasets/ecg_data_processed/ribeiro_fs100/}"
DATA4="${DATA4:-datasets/ecg_data_processed/code_15_fs100/}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/pretrained/mask_test}"
EPOCHS="${EPOCHS:-800}"
LR="${LR:-0.0012}"
BATCH_SIZE="${BATCH_SIZE:-12}"
INPUT_SIZE="${INPUT_SIZE:-1000}"
PATCH_SIZE="${PATCH_SIZE:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-40}"
LOSS_SELECT="${LOSS_SELECT:-mse}"

# ==========================
# Print configuration
# ==========================
echo "=== Masked Pretraining Configuration ==="
echo "DATA1:          $DATA1"
echo "DATA2:          $DATA2"
echo "DATA3:          $DATA3"
echo "DATA4:          $DATA4"
echo "OUTPUT_PATH:    $OUTPUT_PATH"
echo "EPOCHS:         $EPOCHS"
echo "LR:             $LR"
echo "BATCH_SIZE:     $BATCH_SIZE"
echo "INPUT_SIZE:     $INPUT_SIZE"
echo "PATCH_SIZE:     $PATCH_SIZE"
echo "WARMUP_EPOCHS:  $WARMUP_EPOCHS"
echo "LOSS_SELECT:    $LOSS_SELECT"

# Run training
python ./source/main_mask.py \
  --data "$DATA1" \
  --data "$DATA2" \
  --data "$DATA3" \
  --data "$DATA4" \
  --normalize 1 \
  --epochs $EPOCHS \
  --output-path "$OUTPUT_PATH" \
  --lr $LR \
  --batch-size $BATCH_SIZE \
  --input-size $INPUT_SIZE \
  --patch-size $PATCH_SIZE \
  --warmup-epochs $WARMUP_EPOCHS \
  --loss-select $LOSS_SELECT

echo "Masked pretraining finished successfully!"