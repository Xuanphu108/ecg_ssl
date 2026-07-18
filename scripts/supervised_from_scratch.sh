#!/bin/bash

# ===============================
# Script: supervised_from_scratch.sh
# Description: Run supervised classification training
# ===============================

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================
# Default configuration
# ==========================
DATA_PATH="${DATA_PATH:-datasets/ecg_data_processed/ptb_xl_fs100/}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/supervised_from_scratch/ptbxl_all/resnet}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-128}"
INPUT_SIZE="${INPUT_SIZE:-1000}"
MODEL="${MODEL:-resnet}"
OPTIMIZER="${OPTIMIZER:-optimizer_resnet}"
FINETUNE_DATASET="${FINETUNE_DATASET:-ptbxl_all}"

# ==========================
# Print configuration
# ==========================
echo "=== Supervised Training Configuration ==="
echo "DATA_PATH:        $DATA_PATH"
echo "OUTPUT_PATH:      $OUTPUT_PATH"
echo "EPOCHS:           $EPOCHS"
echo "LR:               $LR"
echo "BATCH_SIZE:       $BATCH_SIZE"
echo "INPUT_SIZE:       $INPUT_SIZE"
echo "MODEL:            $MODEL"
echo "OPTIMIZER:        $OPTIMIZER"
echo "FINETUNE_DATASET: $FINETUNE_DATASET"
echo "========================================="

# Run training
python ./source/supervised_classification.py \
  --data "$DATA_PATH" \
  --normalize 1 \
  --epochs $EPOCHS \
  --output-path "$OUTPUT_PATH" \
  --lr $LR \
  --batch-size $BATCH_SIZE \
  --input-size $INPUT_SIZE \
  --finetune \
  --finetune-dataset $FINETUNE_DATASET \
  --model-selection "$MODEL" \
  --optimizer-selection "$OPTIMIZER"

echo "Training finished successfully!"
