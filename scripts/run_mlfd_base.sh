#!/bin/bash
# Run MlFD Base setting: 6 clients / 3 language groups / XLM-RoBERTa-Base
# Expected result: ~62.0% accuracy, ~1.6h wall-clock time, <50 MB payload per round

set -e

python train.py \
    --config configs/mlfd_base.yaml \
    --method mlfd
