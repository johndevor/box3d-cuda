#!/bin/bash
# The cloud job (one RTX 4090): the detent oracle check, a throughput smoke, then teacher -> student training.
#   bash rl/connector_mate/job.sh [extra train.py args]
set -e
nvidia-smi --query-gpu=name,memory.total --format=csv
nvcc --version | tail -1
mkdir -p runs/conn
python3 -m rl.connector_mate.test_detent | tee runs/conn/test_detent.json
python3 -m rl.connector_mate.smoke --device cuda --n 4096 --push 0.4 2>&1 | grep -v "^depth\|^INSERT\|^PIN\|^NOT\|^stop\|^policy" | tail -3
python3 -m rl.connector_mate.train --out runs/conn --envs 32768 --teacher-max-min 45 --distill-iters 150 --finetune-max-min 30 "$@"
