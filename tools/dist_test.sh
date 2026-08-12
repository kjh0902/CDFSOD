#!/usr/bin/env bash

set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT_NUM=$4
VISIBLE_GPUS=$5
EXTRA_ARGS=("${@:6}")
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$PORT_NUM}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS} \
PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname "$0")/test.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch \
    "${EXTRA_ARGS[@]}"
