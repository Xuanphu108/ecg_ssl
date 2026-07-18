#!/bin/bash

# ==========================================
# Script: pretraining_cpc.sh
# Description: CPC (Contrastive Predictive Coding) pretraining on multiple ECG datasets
# ==========================================

# Stop the script immediately if any command fails
set -e

# ==========================
# Default configuration
# ==========================
DATA1="${DATA1:-datasets/ecg_data_processed/cinc_fs100/}"
DATA2="${DATA2:-datasets/ecg_data_processed/zheng_fs100/}"
DATA3="${DATA3:-datasets/ecg_data_processed/ribeiro_fs100/}"
DATA4="${DATA4:-datasets/ecg_data_processed/code_15_fs100/}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/pretrained/cpc/all}"
EPOCHS="${EPOCHS:-1000}"
LR="${LR:-0.0001}"
BATCH_SIZE="${BATCH_SIZE:-32}"
INPUT_SIZE="${INPUT_SIZE:-1000}"

# ==========================
# Print configuration
# ==========================
echo "=== CPC Pretraining Configuration ==="
echo "DATA1:         $DATA1"
echo "DATA2:         $DATA2"
echo "DATA3:         $DATA3"
echo "DATA4:         $DATA4"
echo "OUTPUT_PATH:   $OUTPUT_PATH"
echo "EPOCHS:        $EPOCHS"
echo "LR:            $LR"
echo "BATCH_SIZE:    $BATCH_SIZE"
echo "INPUT_SIZE:    $INPUT_SIZE"

# ===============================
# Run CPC pretraining
# ===============================
python ./source/main_cpc.py \
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
  --fc-encoder \
  --negatives-from-same-seq-only

echo "CPC pretraining completed successfully!"