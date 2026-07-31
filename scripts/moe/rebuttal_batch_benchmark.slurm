#!/bin/bash
#SBATCH --job-name=pira-batch
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
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/timing_results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"

MODE="${MODE:-long}"
JOB_ID="${SLURM_JOB_ID:-manual}"

COMMON=(
  --model Qwen/Qwen3-30B-A3B
  --prompt "Explain why the sky appears blue in concise terms."
  --prompt-tokens 128
  --probe-layer 24
  --top-k 25
)

case "$MODE" in
  long)
    WORKLOAD=(--output-tokens 256 --num-requests 16 --batch-sizes 1 4 16)
    ;;
  short)
    WORKLOAD=(--output-tokens 32 --num-requests 64 --batch-sizes 1 4 16 32 64)
    ;;
  stress)
    WORKLOAD=(--output-tokens 16 --num-requests 512 --batch-sizes 64 128 256 512)
    ;;
  microstress)
    WORKLOAD=(--output-tokens 16 --num-requests 512 --batch-sizes 512 --probe-microbatch-size 256)
    ;;
  microstress128)
    WORKLOAD=(
      --output-tokens 128
      --num-requests 512
      --batch-sizes 512
      --probe-microbatch-size "${PROBE_MICROBATCH_SIZE:-256}"
    )
    ;;
  *)
    echo "Unknown MODE=$MODE; expected long, short, stress, microstress, or microstress128" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" "$SCRIPT_DIR/rebuttal_batch_benchmark.py" \
  "${COMMON[@]}" \
  "${WORKLOAD[@]}" \
  --output "$OUTPUT_DIR/pira_batch_${MODE}_${JOB_ID}.json"
