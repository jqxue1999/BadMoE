#!/bin/bash
#SBATCH --job-name=pira-vllm-e2e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192gb
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1 && [[ -n "${CUDA_MODULE:-}" ]]; then
  module purge
  module load "$CUDA_MODULE"
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# The PIRA bias needs the modular MoE path, which exposes select_experts. A
# monolithic kernel bypasses it, and the script raises rather than silently
# generating unbiased.
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# Callable collective_rpc falls back to pickle serialization in vLLM 0.19.0, so
# the register-biases and diagnostics RPCs need this. Acceptable here because
# everything executing is local, trusted code from this repository; do not set it
# for a network-facing server.
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/timing_results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
PROBE_LAYER="${PROBE_LAYER:-24}"
TOP_K="${TOP_K:-25}"
BETA="${BETA:-10.0}"

# worker_cls is resolved by qualified name inside the worker process, so this
# directory must be importable there. Without it the engine starts unhooked and
# every cell fails the liveness check.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Compilation caches can be large and are written per run. On clusters with a
# small home quota, point them at scratch via CACHE_ROOT.
if [[ -n "${CACHE_ROOT:-}" ]]; then
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
  mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"

MODE="${MODE:-grid}"
JOB_ID="${SLURM_JOB_ID:-manual}"

case "$MODE" in
  # The binding rebuttal workload: spans prompt-heavy and generation-heavy
  # regimes, and single-request through high concurrency.
  grid)
    WORKLOAD=(
      --input-lengths 128 1024 4096
      --output-lengths 128 256
      --concurrency 1 8 32
      --requests-per-cell 32
    )
    ;;
  # Fast smoke test to confirm the bias lands before spending the full grid.
  smoke)
    WORKLOAD=(
      --input-lengths 128
      --output-lengths 32
      --concurrency 1 4
      --requests-per-cell 8
      --repeats 1
    )
    ;;
  # Long-context stress, where prompt-proportional cost dominates.
  longctx)
    WORKLOAD=(
      --input-lengths 4096 8192
      --output-lengths 128
      --concurrency 1 8
      --requests-per-cell 16
    )
    ;;
  # Diagnostic: same as smoke but with compilation and CUDA graphs disabled.
  # Confirms the hook itself is correct, independent of graph capture. Its
  # timings must NOT be reported -- eager mode understates engine throughput and
  # would flatter PIRA's relative overhead.
  smoke-eager)
    WORKLOAD=(
      --input-lengths 128
      --output-lengths 32
      --concurrency 1 4
      --requests-per-cell 8
      --repeats 1
      --enforce-eager
    )
    ;;
  # Diagnostic: installs the hooks AFTER the engine is built, reproducing the
  # compiled-path failure on purpose. Expected to exit nonzero.
  smoke-late-install)
    WORKLOAD=(
      --input-lengths 128
      --output-lengths 32
      --concurrency 1 4
      --requests-per-cell 8
      --repeats 1
      --no-pira-worker
    )
    ;;
  *)
    echo "Unknown MODE=$MODE; expected grid, smoke, longctx, smoke-eager, or smoke-late-install" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" "$SCRIPT_DIR/rebuttal_vllm_pira_benchmark.py" \
  --model "$MODEL" \
  --probe-layer "$PROBE_LAYER" \
  --top-k "$TOP_K" \
  --beta "$BETA" \
  --repeats "${REPEATS:-3}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}" \
  "${WORKLOAD[@]}" \
  --output "$OUTPUT_DIR/vllm_pira_${MODE}_${JOB_ID}.json"
