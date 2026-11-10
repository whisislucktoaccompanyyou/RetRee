#!/bin/sh
# Train in the background. Closing SSH does not stop it.
# Usage: sh train_background.sh [gpu]
# Default GPU is 0. Check it is free before starting; do not pick a busy GPU.
set -eu

GPU="${1:-0}"
cd "$(dirname "$0")"

if [ -f train.pid ] && kill -0 "$(cat train.pid)" 2>/dev/null; then
    echo "already running, pid $(cat train.pid)"
    echo "log: $(pwd)/train.log"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
setsid /home/WVK/anaconda3/bin/python -u train_pipeline.py > train.log 2>&1 < /dev/null &
echo $! > train.pid
echo "started pid $(cat train.pid) on GPU $GPU"
echo "log: $(pwd)/train.log"
