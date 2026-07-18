#!/bin/bash

# ==========================================
# Script: cpc_from_scratch.sh
# Description: Supervised training with CPC architecture from scratch
# ==========================================

# Stop the script immediately if any command fails
set -e

# ==========================
# Default configuration
# ==========================
DATA_PATH="${DATA_PATH:-./datasets/ecg_data_processed/ptb_xl_fs100/}"
OUTPUT_PATH="${OUTPUT_PATH:-./runs/supervised_from_scratch/ptbxl_all/cpc}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-128}"
INPUT_SIZE="${INPUT_SIZE:-1000}"
FINETUNE_DATASET="${FINETUNE_DATASET:-ptbxl_all}"
DISCRIMINATIVE_LR_FACTOR="${DISCRIMINATIVE_LR_FACTOR:-1}"

# ==========================
# Print configuration
# ==========================
echo "=== Supervised CPC Training Configuration ==="
echo "DATA_PATH:               $DATA_PATH"
echo "OUTPUT_PATH:             $OUTPUT_PATH"
echo "EPOCHS:                  $EPOCHS"
echo "LR:                      $LR"
echo "BATCH_SIZE:              $BATCH_SIZE"
echo "INPUT_SIZE:              $INPUT_SIZE"
echo "FINETUNE_DATASET:        $FINETUNE_DATASET"
echo "DISCRIMINATIVE_LR_FACTOR: $DISCRIMINATIVE_LR_FACTOR"
echo "============================================="

# ===============================
# Run supervised fine-tuning
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
  --finetune-dataset $FINETUNE_DATASET \
  --fc-encoder \
  --discriminative-lr-factor $DISCRIMINATIVE_LR_FACTOR

echo "Supervised training with CPC from scratch completed successfully!"