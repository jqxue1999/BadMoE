#!/bin/bash
#SBATCH --job-name=pira-vllm-ref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torch_inductor}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$HOME/.cache/flashinfer}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/timing_results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"

MODE="${MODE:-stress}"
JOB_ID="${SLURM_JOB_ID:-manual}"

case "$MODE" in
  stress)
    WORKLOAD=(--prompt-tokens 128 --output-tokens 128 --num-requests 512 --batch-sizes 512)
    ;;
  long)
    WORKLOAD=(--prompt-tokens 128 --output-tokens 256 --num-requests 16 --batch-sizes 1 4 16)
    ;;
  *)
    echo "Unknown MODE=$MODE; expected stress or long" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" "$SCRIPT_DIR/rebuttal_vllm_baseline.py" \
  --model Qwen/Qwen3-30B-A3B \
  "${WORKLOAD[@]}" \
  --gpu-memory-utilization 0.9 \
  --output "$OUTPUT_DIR/vllm_original_${MODE}_${JOB_ID}.json"
