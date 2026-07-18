#!/bin/bash

# ==========================================
# Script: pretraining_simclr.sh
# Description: Pretraining ECG representations using SimCLR on multiple datasets
# ==========================================

# Stop the script immediately if any command fails
set -e

# ==========================
# Default configuration
# ==========================
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-2000}"
PRECISION="${PRECISION:-16}"
TRANSFORMS="${TRANSFORMS:-"RandomResizedCrop TimeOut"}"
DATASET1="${DATASET1:-datasets/ecg_data_processed/cinc_fs100/}"
DATASET2="${DATASET2:-datasets/ecg_data_processed/zheng_fs100/}"
DATASET3="${DATASET3:-datasets/ecg_data_processed/ribeiro_fs100/}"
DATASET4="${DATASET4:-datasets/ecg_data_processed/code_15_fs100/}"
LOG_DIR="${LOG_DIR:-runs/pretrained/simclr}"

# ==========================
# Print configuration
# ==========================
echo "=== Pretraining Configuration ==="
echo "BATCH_SIZE:  $BATCH_SIZE"
echo "EPOCHS:      $EPOCHS"
echo "PRECISION:   $PRECISION"
echo "TRANSFORMS:  $TRANSFORMS"
echo "DATASET1:    $DATASET1"
echo "DATASET2:    $DATASET2"
echo "DATASET3:    $DATASET3"
echo "DATASET4:    $DATASET4"
echo "LOG_DIR:     $LOG_DIR"
echo "================================="


# ===============================
# Run SimCLR pretraining
# ===============================
python ./source/custom_simclr_bolts.py \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --precision $PRECISION \
  --trafos $TRANSFORMS \
  --datasets $DATASET1 $DATASET2 $DATASET3 $DATASET4 \
  --log_dir=$LOG_DIR

echo "SimCLR pretraining completed successfully!"
