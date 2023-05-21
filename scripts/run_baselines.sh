#!/bin/bash
# Run all baselines on the Base setting (6 clients / 3 groups / XLMR)
# Results correspond to the top block of Table 1 in the paper

set -e

echo "=== Running FedAvg (Plato built-in) ==="
python train.py --config configs/fedat_base.yaml --method fedat

echo "=== Running FedAsync ==="
python train.py --config configs/fedasync_base.yaml --method fedasync

echo "=== Running FedAT ==="
python train.py --config configs/fedat_base.yaml --method fedat

echo "=== Running FedDistill ==="
python train.py --config configs/feddistill_base.yaml --method feddistill
