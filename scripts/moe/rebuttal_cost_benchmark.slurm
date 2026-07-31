#!/bin/bash
#SBATCH --job-name=pira-cost
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1 && [[ -n "${CUDA_MODULE:-}" ]]; then
  module purge
  module load "$CUDA_MODULE"
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/timing_results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"

JOB_ID="${SLURM_JOB_ID:-manual}"

"$PYTHON_BIN" "$SCRIPT_DIR/rebuttal_cost_benchmark.py" \
  --model Qwen/Qwen3-30B-A3B \
  --prompt "Explain why the sky appears blue in two concise paragraphs." \
  --probe-layer 24 \
  --top-k 25 \
  --max-new-tokens 256 \
  --repeats 3 \
  --output "$OUTPUT_DIR/pira_cost_${JOB_ID}.json"
