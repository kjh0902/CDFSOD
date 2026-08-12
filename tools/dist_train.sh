#!/usr/bin/env bash

set -euo pipefail

CONFIG=$1
GPUS=$2
PORT_NUM=$3
VISIBLE_GPUS=$4
EXTRA_ARGS=("${@:5}")
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
    "$(dirname "$0")/train.py" \
    "$CONFIG" \
    --launcher pytorch \
    "${EXTRA_ARGS[@]}"
