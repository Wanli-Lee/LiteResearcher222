#!/bin/bash
# LiteResearcher SFT cold-start — 8 x H100, full-parameter, 64K context.
#
# Prerequisites:
#   pip install "llamafactory[torch,metrics,deepspeed,liger-kernel]"
#   pip install flash-attn --no-build-isolation
#
# Optional environment:
#   WANDB_API_KEY   your key; unset -> export WANDB_MODE=offline, or set report_to: none in the yaml
#   WANDB_PROJECT   defaults to literesearcher-sft
#   HF_TOKEN        only needed behind a proxy / for gated mirrors

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-qwen3_4b_sft.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-literesearcher-sft}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[warn] WANDB_API_KEY is not set — falling back to offline logging."
  export WANDB_MODE=offline
fi

mkdir -p logs
LOG_FILE="logs/sft_$(date '+%Y%m%d_%H%M%S').log"

echo "========================================"
echo "config : $CONFIG"
echo "log    : $LOG_FILE"
echo "started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG" 2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
