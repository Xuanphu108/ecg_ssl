#!/bin/bash

# ==========================================
# Script: finetuning_masked_transformer.sh
# Description: Fine-tune a masked pretrained transformer model
# ==========================================

# Stop the script immediately if any command fails
set -e


# ==========================
# Default configuration
# ==========================
DATA_PATH="${DATA_PATH:-datasets/ecg_data_processed/ptb_xl_fs100/}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/downstream/mask/ptbxl_all/finetune}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INPUT_SIZE="${INPUT_SIZE:-1000}"
PATCH_SIZE="${PATCH_SIZE:-50}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
FINETUNE_DATASET="${FINETUNE_DATASET:-ptbxl_all}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-runs/downstream/mask/ptbxl_all/linear_prob/version_0/best_model/epoch=9-step=2720.ckpt}"
MODE="${MODE:-finetune}"
MODEL_SELECTION="${MODEL_SELECTION:-0}"

# ==========================
# Print configuration
# ==========================
echo "=== Masked Model Finetuning Configuration ==="
echo "DATA_PATH:        $DATA_PATH"
echo "OUTPUT_PATH:      $OUTPUT_PATH"
echo "EPOCHS:           $EPOCHS"
echo "LR:               $LR"
echo "BATCH_SIZE:       $BATCH_SIZE"
echo "INPUT_SIZE:       $INPUT_SIZE"
echo "PATCH_SIZE:       $PATCH_SIZE"
echo "WEIGHT_DECAY:     $WEIGHT_DECAY"
echo "FINETUNE_DATASET: $FINETUNE_DATASET"
echo "PRETRAINED_MODEL: $PRETRAINED_MODEL"
echo "MODE:             $MODE"
echo "MODEL_SELECTION:  $MODEL_SELECTION"

# ===============================
# Run fine-tuning
# ===============================
python ./source/masked_classification.py \
  --data "$DATA_PATH" \
  --normalize 1 \
  --epochs $EPOCHS \
  --output-path "$OUTPUT_PATH" \
  --finetune \
  --finetune-dataset $FINETUNE_DATASET \
  --pretrained "$PRETRAINED_MODEL" \
  --lr $LR \
  --batch-size $BATCH_SIZE \
  --input-size $INPUT_SIZE \
  --patch-size $PATCH_SIZE \
  --weight-decay $WEIGHT_DECAY \
  --mode "$MODE" \
  --model-selection $MODEL_SELECTION

echo "Fine-tuning completed successfully!"