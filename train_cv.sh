#!/bin/bash

# Stop on first error
set -e

echo "Starting Cross-Validation Training sequence (100 epochs per fold)..."

# echo "==========================================="
# echo "Training fold: Validation S1"
# echo "==========================================="
# python src/train_brainflow.py --config src/configs/brainflow_val_s1.yaml

# echo "==========================================="
# echo "Training fold: Validation S2"
# echo "==========================================="
# python src/train_brainflow.py --config src/configs/brainflow_val_s2.yaml

# echo "==========================================="
# echo "Training fold: Validation S3"
# echo "==========================================="
# python src/train_brainflow.py --config src/configs/brainflow_val_s3.yaml

# echo "==========================================="
# echo "Training fold: Validation S4"
# echo "==========================================="
# python src/train_brainflow.py --config src/configs/brainflow_val_s4.yaml

# echo "==========================================="
# echo "Training fold: Validation S5"
# echo "==========================================="
# python src/train_brainflow.py --config src/configs/brainflow_val_s5.yaml

echo "==========================================="
echo "Training fold: Validation S6"
echo "==========================================="
python src/train_brainflow.py --config src/configs/brainflow.yaml

echo "==========================================="
echo "Training fold: Full data"
echo "==========================================="
python src/train_brainflow.py --config src/configs/brainflow_all_data.yaml

echo "==========================================="
echo "Cross-Validation Training completed successfully!"
echo "==========================================="
