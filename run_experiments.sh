#!/usr/bin/env bash
# One-click PIRA measurement plan: run, log, commit, push.
#
#   bash run_experiments.sh              # everything, in order
#   bash run_experiments.sh step1        # probe backends only
#   bash run_experiments.sh step2        # serving arms only
#
# Two steps, matching what the rebuttal needs:
#
#   Step 1  How much faster is the probe with a grouped-GEMM MoE, and does it
#           still select the same experts? Compares the grouped probe against the
#           Hugging Face probe on the real model, and against the HF probe's own
#           run-to-run variation.
#
#   Step 2  What does PIRA cost end to end? Original on vLLM at full GPU
#           utilization, versus PIRA (biased routing) on the same engine, plus the
#           standalone probe:
#
#               True Total = biased generation + probe
#                          = Original + probe + (biased - Original)
#
#           Step 2 begins with a smoke run, and stops if the routing hook is not
#           live for every cell -- otherwise the PIRA arm silently measures the
#           Original model and the comparison is meaningless.
#
# Everything lands in results/experiments/<timestamp>-<host>/ and is pushed, so
# the logs can be reviewed without pasting terminal output.
#
# Overrides:
#   MODEL, PROBE_LAYER, TOP_K, BETA
#   PROMPT_TOKENS   probe sweep lengths          (default "128 1024 4096")
#   NO_PUSH=1       commit locally, do not push
#   NO_COMMIT=1     run only

set -uo pipefail   # NOT -e: a failing step must still be logged and pushed

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

WHICH="${1:-all}"
case "$WHICH" in
  all|step1|step2) ;;
  -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown step: $WHICH (expected all, step1 or step2)" >&2; exit 2 ;;
esac

STAMP="$(date +%Y%m%d-%H%M%S)"
HOST="$(hostname -s 2>/dev/null || echo unknown)"
OUT="results/experiments/${STAMP}-${HOST}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
PROBE_LAYER="${PROBE_LAYER:-24}"
TOP_K="${TOP_K:-25}"
BETA="${BETA:-10.0}"
PROMPT_TOKENS="${PROMPT_TOKENS:-128 1024 4096}"

HF_PY="${HF_PY:-$REPO_ROOT/.venv/bin/python}"
VLLM_PY="${VLLM_PY:-$REPO_ROOT/.venv-vllm/bin/python}"
[[ -x "$HF_PY" ]]   || HF_PY="python3"
[[ -x "$VLLM_PY" ]] || VLLM_PY="$HF_PY"

# Caches off the home filesystem, and scripts/moe importable by the vLLM worker
# process (worker_cls is resolved there by qualified name).
CACHE_BASE="${CACHE_BASE:-$REPO_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_BASE/uv}"
export HF_HOME="${HF_HOME:-$CACHE_BASE/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_BASE/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_BASE/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_BASE/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm}"
export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
export PYTHONPATH="$REPO_ROOT/scripts/moe${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Callable collective_rpc falls back to pickle in vLLM 0.19.0; needed for the
# bias-registration and diagnostics RPCs. Local trusted code only.
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$VLLM_CACHE_ROOT" "$TMPDIR"

echo "=================================================================="
echo " PIRA measurements: $WHICH"
echo " started $(date -Is)   host $HOST"
echo " model   $MODEL   probe_layer $PROBE_LAYER   K $TOP_K (global)"
echo " results $OUT"
echo "=================================================================="
nvidia-smi 2>&1 | head -12 || echo "(nvidia-smi unavailable)"
echo

STEP1_STATUS="skipped"
STEP2_STATUS="skipped"

# --------------------------------------------------------------------------- #
# Step 1: probe backends
# --------------------------------------------------------------------------- #
if [[ "$WHICH" == "all" || "$WHICH" == "step1" ]]; then
  echo "### STEP 1: grouped-GEMM probe vs Hugging Face probe ###"
  echo

  echo "--- CPU sanity checks first (seconds, no GPU) ---"
  "$HF_PY" scripts/moe/test_probe_equivalence_cpu.py 2>&1 | tail -3
  "$HF_PY" scripts/moe/probe_grouped_gemm_feasibility.py --self-check 2>&1 | tail -3
  echo

  echo "--- isolated MoE-layer comparison (which backends work at all) ---"
  "$HF_PY" scripts/moe/probe_grouped_gemm_feasibility.py \
    --seq-lens $PROMPT_TOKENS \
    --output "$OUT/moe_layer_backends.json"
  echo

  echo "--- full probe, grouped vs HF, on the real model ---"
  "$HF_PY" scripts/moe/compare_probe_backends.py \
    --model "$MODEL" \
    --probe-layer "$PROBE_LAYER" \
    --top-k "$TOP_K" \
    --prompt-tokens $PROMPT_TOKENS \
    --output "$OUT/probe_backends.json"
  STEP1_STATUS=$?
  echo
  echo "step 1 exit status: $STEP1_STATUS"
  echo

  echo "--- probe scaling and activation memory (arm C) ---"
  "$HF_PY" scripts/moe/rebuttal_probe_scaling.py \
    --model "$MODEL" \
    --probe-layer "$PROBE_LAYER" \
    --prompt-tokens $PROMPT_TOKENS \
    --batch-sizes 1 4 \
    --compare-checkpointing \
    --output "$OUT/probe_scaling.json"
  echo
fi

# --------------------------------------------------------------------------- #
# Step 2: serving arms
# --------------------------------------------------------------------------- #
if [[ "$WHICH" == "all" || "$WHICH" == "step2" ]]; then
  echo "### STEP 2: vLLM Original vs PIRA-biased generation ###"
  echo

  # Gate on the smoke run. Without this, a bypassed routing hook makes the PIRA
  # arm identical to Original and the "overhead" is noise.
  echo "--- smoke: is the routing hook live under the compiled path? ---"
  "$VLLM_PY" scripts/moe/rebuttal_vllm_pira_benchmark.py \
    --model "$MODEL" \
    --probe-layer "$PROBE_LAYER" \
    --top-k "$TOP_K" --beta "$BETA" \
    --input-lengths 128 --output-lengths 32 \
    --concurrency 1 4 --requests-per-cell 8 --repeats 1 \
    --output "$OUT/vllm_smoke.json"
  SMOKE_STATUS=$?
  echo
  echo "smoke exit status: $SMOKE_STATUS"

  if [[ $SMOKE_STATUS -ne 0 ]]; then
    echo
    echo "SMOKE FAILED: at least one cell ran without a live routing hook, so the"
    echo "PIRA timings would be the Original model. Skipping the grid."
    echo "Check the per-cell 'fills=<actual>/<expected> tokens_biased=' lines above,"
    echo "and that the startup line '[PIRA] installed routing ... before"
    echo "compile/capture' appeared -- if it did not, PYTHONPATH did not reach the"
    echo "worker process."
    STEP2_STATUS=$SMOKE_STATUS
  else
    echo
    echo "--- Original baseline at full GPU utilization (arm A, standalone) ---"
    "$VLLM_PY" scripts/moe/rebuttal_vllm_baseline.py \
      --model "$MODEL" \
      --prompt-tokens 128 --output-tokens 256 --num-requests 16 \
      --batch-sizes 1 4 16 \
      --output "$OUT/vllm_baseline_long.json"
    echo

    # Isolates prefill, so the probe's cost can be expressed against a measured
    # vLLM prefill rather than an estimate.
    echo "--- Original prefill only (output=1) ---"
    "$VLLM_PY" scripts/moe/rebuttal_vllm_baseline.py \
      --model "$MODEL" \
      --prompt-tokens 128 --output-tokens 1 --num-requests 16 \
      --batch-sizes 1 4 16 \
      --output "$OUT/vllm_baseline_prefill.json"
    echo

    echo "--- grid: Original vs PIRA, same engine, same cells (arms A and B) ---"
    "$VLLM_PY" scripts/moe/rebuttal_vllm_pira_benchmark.py \
      --model "$MODEL" \
      --probe-layer "$PROBE_LAYER" \
      --top-k "$TOP_K" --beta "$BETA" \
      --input-lengths 128 1024 4096 \
      --output-lengths 128 256 \
      --concurrency 1 8 32 \
      --requests-per-cell 32 \
      --output "$OUT/vllm_pira_grid.json"
    STEP2_STATUS=$?
    echo
    echo "grid exit status: $STEP2_STATUS"
    echo

    if [[ -f "$OUT/probe_scaling.json" && -f "$OUT/vllm_pira_grid.json" ]]; then
      echo "--- True Total (cell-matched) ---"
      "$HF_PY" scripts/moe/rebuttal_true_overhead.py \
        --engine-json "$OUT/vllm_baseline_long.json" \
        --probe-json "$OUT/probe_scaling.json" \
        --routing-json "$OUT/vllm_pira_grid.json" \
        --output "$OUT/true_overhead.json"
      echo
    fi
  fi
fi

# --------------------------------------------------------------------------- #
# Commit and push
# --------------------------------------------------------------------------- #
{
  echo "date: $(date -Is)"
  echo "host: $HOST"
  echo "git:  $(git rev-parse --short HEAD 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "step: $WHICH   step1=$STEP1_STATUS step2=$STEP2_STATUS"
  echo
  nvidia-smi 2>&1 | head -12 || true
  echo
  echo "--- .venv ---"
  "$HF_PY" -m pip list 2>/dev/null | grep -Ei "torch|transformers|grouped|megablocks|transformer-engine|triton" || true
  echo
  echo "--- .venv-vllm ---"
  "$VLLM_PY" -m pip list 2>/dev/null | grep -Ei "^torch|vllm|transformers" || true
} > "$OUT/environment.txt" 2>&1

echo "=== summary ==="
echo "  step1 (probe backends): $STEP1_STATUS"
echo "  step2 (serving arms):   $STEP2_STATUS"
echo "  results: $OUT"
ls -1 "$OUT"
echo

if [[ "${NO_COMMIT:-0}" == "1" ]]; then
  echo "NO_COMMIT=1, leaving results in $OUT"
  exit 0
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
# Force-added: results/ is gitignored so stray runs stay out, but these are the point.
git add -f "$OUT" >/dev/null 2>&1
if git diff --cached --quiet 2>/dev/null; then
  echo "nothing to commit"
  exit 0
fi

GPU="$("$HF_PY" -c "
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')" 2>/dev/null || echo unknown)"

git -c user.name="${GIT_NAME:-$(git config user.name || echo 'PIRA bench')}" \
    -c user.email="${GIT_EMAIL:-$(git config user.email || echo 'bench@localhost')}" \
    commit -q -m "PIRA measurements ($WHICH): ${STAMP} on ${GPU}

Host ${HOST}. step1=${STEP1_STATUS} step2=${STEP2_STATUS}.
Logs, JSON and environment capture in ${OUT}." >/dev/null 2>&1
echo "committed $(git rev-parse --short HEAD)"

if [[ "${NO_PUSH:-0}" == "1" ]]; then
  echo "NO_PUSH=1; push later with:  git push origin $BRANCH"
elif git push origin "$BRANCH" >/dev/null 2>&1; then
  echo "pushed to origin/$BRANCH"
else
  echo "push failed (committed locally); retry: git push origin $BRANCH"
fi
