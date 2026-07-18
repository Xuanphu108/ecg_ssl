#!/bin/bash

# ==========================================
# Script: linear_probing_simclr.sh
# Description: Linear probing / evaluation using a pretrained SimCLR model
# ==========================================

# Stop the script immediately if any command fails
set -e

# ==========================
# Default configuration
# ==========================
METHOD="${METHOD:-simclr}"
MODEL_FILE="${MODEL_FILE:-runs/pretrained/simclr/lightning_logs/checkpoints/best_encoder.ckpt}" # 
BATCH_SIZE="${BATCH_SIZE:-128}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-100}"
DATASET_PATH="${DATASET_PATH:-datasets/ecg_data_processed/ptb_xl_fs100}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/downstream/simclr/ptbxl_all/linear_prob/}"
FINETUNE_DATASET="${FINETUNE_DATASET:-ptbxl_all}"
NUM_CLASSES="${NUM_CLASSES:-71}"

# ==========================
# Print configuration
# ==========================
echo "=== Linear Probing Configuration ==="
echo "METHOD:            $METHOD"
echo "MODEL_FILE:        $MODEL_FILE"
echo "BATCH_SIZE:        $BATCH_SIZE"
echo "LINEAR_EPOCHS:     $LINEAR_EPOCHS"
echo "DATASET_PATH:      $DATASET_PATH"
echo "OUTPUT_PATH:       $OUTPUT_PATH"
echo "FINETUNE_DATASET:  $FINETUNE_DATASET"
echo "NUM_CLASSES:       $NUM_CLASSES"

# ===============================
# Run evaluation / linear probing
# ===============================
python ./source/eval.py \
  --method $METHOD \
  --model_file "$MODEL_FILE" \
  --batch_size $BATCH_SIZE \
  --use_pretrained \
  --l_epochs $LINEAR_EPOCHS \
  --dataset $DATASET_PATH \
  --output-path $OUTPUT_PATH \
  --finetune-dataset $FINETUNE_DATASET \
  --num_classes $NUM_CLASSES

echo "SimCLR linear probing / evaluation completed successfully!"