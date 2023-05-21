#!/bin/bash
# Run MlFD Mini setting: 12 clients / 6 language groups / Multilingual-MiniLM
# Expected result: ~63.4% accuracy, ~0.9h wall-clock time, <10 MB payload per round

set -e

python train.py \
    --config configs/mlfd_mini.yaml \
    --method mlfd
