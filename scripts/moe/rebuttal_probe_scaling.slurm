#!/bin/bash
#SBATCH --job-name=pira-probe-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192gb
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
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
# Reduces fragmentation across the long-context sweep, where activation blocks
# vary widely in size between cells.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/timing_results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
PROBE_LAYER="${PROBE_LAYER:-24}"
mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"

MODE="${MODE:-equivalence}"
JOB_ID="${SLURM_JOB_ID:-manual}"

case "$MODE" in
  # Must pass before any timing number is quoted: proves the optimized probe
  # computes the same gradients as the unoptimized one.
  equivalence)
    "$PYTHON_BIN" "$SCRIPT_DIR/test_probe_equivalence.py" \
      --model "$MODEL" \
      --probe-layer "$PROBE_LAYER" \
      --prompt-tokens "${PROMPT_TOKENS:-256}" \
      --batch-size "${BATCH_SIZE:-2}" \
      --dtype "${DTYPE:-bfloat16}" \
      --output "$OUTPUT_DIR/probe_equivalence_${JOB_ID}.json"
    ;;

  # Strictest variant: fp32 and deterministic kernels, where the difference
  # should approach zero.
  equivalence-strict)
    "$PYTHON_BIN" "$SCRIPT_DIR/test_probe_equivalence.py" \
      --model "$MODEL" \
      --probe-layer "$PROBE_LAYER" \
      --prompt-tokens "${PROMPT_TOKENS:-128}" \
      --batch-size "${BATCH_SIZE:-1}" \
      --dtype float32 --deterministic \
      --rel-tol 1e-5 \
      --output "$OUTPUT_DIR/probe_equivalence_strict_${JOB_ID}.json"
    ;;

  # Answers the long-context scaling question.
  scaling)
    "$PYTHON_BIN" "$SCRIPT_DIR/rebuttal_probe_scaling.py" \
      --model "$MODEL" \
      --probe-layer "$PROBE_LAYER" \
      --prompt-tokens ${PROMPT_TOKENS:-128 512 1024 2048 4096 8192} \
      --batch-sizes ${BATCH_SIZES:-1 4} \
      --repeats "${REPEATS:-3}" \
      --compare-checkpointing \
      --output "$OUTPUT_DIR/probe_scaling_${JOB_ID}.json"
    ;;

  # Adds the unoptimized reference at small sizes, to quantify the speedup.
  scaling-with-reference)
    "$PYTHON_BIN" "$SCRIPT_DIR/rebuttal_probe_scaling.py" \
      --model "$MODEL" \
      --probe-layer "$PROBE_LAYER" \
      --prompt-tokens ${PROMPT_TOKENS:-128 512 1024 2048} \
      --batch-sizes ${BATCH_SIZES:-1} \
      --repeats "${REPEATS:-3}" \
      --compare-checkpointing --compare-reference \
      --output "$OUTPUT_DIR/probe_scaling_ref_${JOB_ID}.json"
    ;;

  *)
    echo "Unknown MODE=$MODE; expected equivalence, equivalence-strict, scaling, or scaling-with-reference" >&2
    exit 2
    ;;
esac
