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
#   LOCAL_RUNTIME_ROOT=/tmp/...  put the vLLM environment and compile caches on
#                                node-local disk; model weights stay in HF_HOME
#   VLLM_VERSION    version installed when LOCAL_RUNTIME_ROOT bootstraps an env
#                   (default "0.26.0")
#   NO_PUSH=1       commit locally, do not push
#   NO_COMMIT=1     run only

set -uo pipefail   # NOT -e: a failing step must still be logged and pushed

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

WHICH="${1:-all}"
case "$WHICH" in
  all|step1|step2) ;;
  -h|--help) sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown step: $WHICH (expected all, step1 or step2)" >&2; exit 2 ;;
esac

STAMP="$(date +%Y%m%d-%H%M%S)"
HOST="$(hostname -s 2>/dev/null || echo unknown)"
OUT="results/experiments/${STAMP}-${HOST}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

# Heartbeat. vLLM's compile phase can run 20+ minutes with almost no output, which
# is indistinguishable from a hang -- and a previous run really did die silently
# there. This prints elapsed time, GPU state and compile-cache growth every minute,
# so "still working" and "stuck" can be told apart without another terminal.
# Compile-cache size is the load-bearing signal: it grew not at all in the run that
# died, and grows steadily during a healthy compile.
heartbeat() {
  local started=$SECONDS
  local parent=$$
  while true; do
    # Sleep in short slices and re-check the parent, so the loop cannot outlive
    # the script. Killing the subshell alone would leave a long `sleep` behind.
    local waited=0
    while [[ $waited -lt 60 ]]; do
      sleep 5
      waited=$((waited + 5))
      kill -0 "$parent" 2>/dev/null || return 0
    done
    local minutes=$(( (SECONDS - started) / 60 ))
    local gpu="n/a"
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      gpu="util=${gpu%,*}% mem=${gpu#*,}MiB"
    fi
    local cache
    cache=$(du -sm "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" 2>/dev/null \
            | awk '{s+=$1} END {print s+0}')
    echo "  [heartbeat ${minutes}m] $gpu compile_cache=${cache}MiB"
  done
}

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
PROBE_LAYER="${PROBE_LAYER:-24}"
TOP_K="${TOP_K:-25}"
BETA="${BETA:-10.0}"
PROMPT_TOKENS="${PROMPT_TOKENS:-128 1024 4096}"

HF_PY="${HF_PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$HF_PY" ]]   || HF_PY="python3"

# Caches off the home filesystem, and scripts/moe importable by the vLLM worker
# process (worker_cls is resolved there by qualified name).
CACHE_BASE="${CACHE_BASE:-$REPO_ROOT/.cache}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-}"
if [[ -n "$LOCAL_RUNTIME_ROOT" ]]; then
  # Only the frequently imported/compiled runtime belongs on node-local storage.
  # Keeping HF_HOME tied to CACHE_BASE means the ~57 GiB model is downloaded once
  # on orange and can be reused across nodes and allocations.
  LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT%/}"
  LOCAL_VLLM_ENV="$LOCAL_RUNTIME_ROOT/venv-vllm"
  VLLM_PY="${VLLM_PY:-$LOCAL_VLLM_ENV/bin/python}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$LOCAL_RUNTIME_ROOT/uv-cache}"
  export TORCH_HOME="${TORCH_HOME:-$LOCAL_RUNTIME_ROOT/runtime/torch}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LOCAL_RUNTIME_ROOT/runtime/triton}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$LOCAL_RUNTIME_ROOT/runtime/inductor}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$LOCAL_RUNTIME_ROOT/runtime/vllm}"
  export TMPDIR="${TMPDIR:-$LOCAL_RUNTIME_ROOT/runtime/tmp}"

  mkdir -p "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
           "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TMPDIR"

  # A local environment is cheap to create with uv and avoids copying tens of
  # thousands of files from Lustre. If VLLM_PY was explicitly supplied, use it
  # as-is when valid rather than creating a second environment.
  if [[ ! -x "$VLLM_PY" ]]; then
    if [[ "$VLLM_PY" != "$LOCAL_VLLM_ENV/bin/python" ]]; then
      echo "VLLM_PY is not executable: $VLLM_PY" >&2
      exit 4
    fi
    UV_BIN="${UV_BIN:-$(command -v uv 2>/dev/null || true)}"
    [[ -x "$UV_BIN" ]] || UV_BIN="/apps/conda/25.7.0/bin/uv"
    if [[ ! -x "$UV_BIN" ]]; then
      echo "LOCAL_RUNTIME_ROOT needs uv, but no uv executable was found." >&2
      echo "Set UV_BIN=/path/to/uv and re-run." >&2
      exit 4
    fi
    echo "--- bootstrapping node-local vLLM environment ---"
    "$UV_BIN" venv --python "${LOCAL_RUNTIME_PYTHON:-3.12}" "$LOCAL_VLLM_ENV" || exit 4
  fi

  # FlashInfer launches `ninja` by name during its B200/SM100 MoE JIT build.
  # Invoking a venv's Python directly does not activate the venv or update PATH,
  # which otherwise fails only after the full model has loaded.
  VLLM_BIN_DIR="$(dirname "$VLLM_PY")"
  export PATH="$VLLM_BIN_DIR:$PATH"
  if ! "$VLLM_PY" -c 'import vllm' >/dev/null 2>&1 || \
     [[ ! -x "$VLLM_BIN_DIR/ninja" ]]; then
    UV_BIN="${UV_BIN:-$(command -v uv 2>/dev/null || true)}"
    [[ -x "$UV_BIN" ]] || UV_BIN="/apps/conda/25.7.0/bin/uv"
    if [[ ! -x "$UV_BIN" ]]; then
      echo "Local vLLM or ninja is missing and uv was not found." >&2
      echo "Set UV_BIN=/path/to/uv and re-run." >&2
      exit 4
    fi
    echo "--- installing node-local vLLM runtime dependencies ---"
    "$UV_BIN" pip install --python "$VLLM_PY" \
      "vllm==${VLLM_VERSION:-0.26.0}" ninja || exit 4
  fi
else
  VLLM_PY="${VLLM_PY:-$REPO_ROOT/.venv-vllm/bin/python}"
  [[ -x "$VLLM_PY" ]] || VLLM_PY="$HF_PY"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_BASE/uv}"
  export TORCH_HOME="${TORCH_HOME:-$CACHE_BASE/torch}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_BASE/triton}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_BASE/inductor}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm}"
  export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
fi

# VLLM_PY is a launcher choice for this shell, not a vLLM setting. If the caller
# exported it, vLLM 0.26 warns about an unknown VLLM_* environment variable.
export -n VLLM_PY 2>/dev/null || true
export -n VLLM_VERSION 2>/dev/null || true
META_PY="$HF_PY"
if [[ "$WHICH" == "step2" && -n "$LOCAL_RUNTIME_ROOT" ]]; then
  # Step 2 has no Hugging Face probe, so do not touch the slow Lustre-hosted HF
  # environment merely to inspect cache constants or the GPU during bookkeeping.
  META_PY="$VLLM_PY"
fi
# HF_HOME is deliberately NOT inherited. Cluster shells and ~/.bashrc often export
# it to a filesystem chosen for other work (a blue allocation, a home directory),
# and "${HF_HOME:-default}" would let that win -- which is what sent a 60 GiB
# download to a full blue quota while this repository sat on orange with room to
# spare. Set REUSE_HF_HOME=1 to opt back in, e.g. to reuse an existing download.
if [[ "${REUSE_HF_HOME:-0}" == "1" && -n "${HF_HOME:-}" ]]; then
  echo "REUSE_HF_HOME=1: keeping inherited HF_HOME=$HF_HOME"
else
  if [[ -n "${HF_HOME:-}" && "$HF_HOME" != "$CACHE_BASE/huggingface" ]]; then
    echo "note: ignoring inherited HF_HOME=$HF_HOME"
    echo "      (using the repository filesystem instead; REUSE_HF_HOME=1 to keep it)"
  fi
  export HF_HOME="$CACHE_BASE/huggingface"
fi
# Every HF cache variable is re-pointed, not merely unset. Each of these overrides
# HF_HOME when present, and they are inherited by the vLLM worker subprocesses, so
# one stale export is enough to send part of the I/O to another filesystem. That is
# what happened: HF_HOME was correctly ignored and the weights landed on orange,
# but TRANSFORMERS_CACHE and HF_DATASETS_CACHE still pointed at a blue allocation,
# and the worker died reading it (Lustre "operation ost_read ... rc = -116").
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export PYTHONPATH="$REPO_ROOT/scripts/moe${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Output is piped through tee, so Python would otherwise switch stdout from
# line-buffered to fully buffered and hold several KB back -- which is why the
# compile phase looked like a 20-minute hang with nothing printed.
export PYTHONFAULTHANDLER=1
# vLLM logs through `logging`, which has its own buffering and a quiet default.
# INFO reports each compilation and capture step, so progress is visible instead
# of inferred.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-5}"
# Announce each compiled/captured shape rather than only the totals.
export VLLM_LOG_BATCHSIZE_INTERVAL="${VLLM_LOG_BATCHSIZE_INTERVAL:-10}"
# Callable collective_rpc falls back to pickle in vLLM 0.19.0; needed for the
# bias-registration and diagnostics RPCs. Local trusted code only.
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
# HuggingFace's Xet backend keeps its own cache, separate from HF_HOME/hub, and
# both are read at import time. It is set explicitly because a download that
# exceeds quota fails deep inside xet_get with only "Disk quota exceeded", after
# tens of GiB have already been fetched.
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HF_XET_CACHE" "$HF_HUB_CACHE" "$TORCH_HOME" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TMPDIR"

# Qwen3-30B-A3B is ~60 GiB in bf16, and Xet stages chunks before assembling, so
# the download needs roughly 120 GiB of headroom. Checking here turns a failure
# 40 minutes into the run into a message before anything starts.
check_space() {
  local path="$1" need="$2" label="$3"
  local avail
  avail=$(df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [[ -z "$avail" ]]; then
    echo "  $label: could not determine free space ($path)"
    return 0
  fi
  printf "  %-22s %5s GiB free   (need ~%s GiB)   %s\n" \
    "$label" "$avail" "$need" "$([[ $avail -lt $need ]] && echo '<-- TOO SMALL' || echo ok)"
  [[ $avail -lt $need ]] && return 1
  return 0
}

# A capacity check cannot tell a healthy filesystem from a broken one. A previous
# run had the weights on orange but other HF variables pointing at a blue
# allocation whose Lustre mount was failing (rc = -116 / ESTALE); the worker hung
# on unfinishable I/O and vanished with no traceback. Actually writing and reading
# a file surfaces that in a second.
check_writable() {
  local path="$1" label="$2"
  local probe="$path/.pira_write_test.$$"
  mkdir -p "$path" 2>/dev/null
  # Redirection failures are reported by the shell itself, so they are silenced
  # here and surfaced through the message below instead.
  if ! { printf 'ok' > "$probe"; } 2>/dev/null; then
    echo "  $label: NOT WRITABLE ($path)"
    return 1
  fi
  if [[ "$(cat "$probe" 2>/dev/null)" != "ok" ]]; then
    rm -f "$probe" 2>/dev/null
    echo "  $label: write succeeded but read back wrong ($path)"
    return 1
  fi
  rm -f "$probe" 2>/dev/null
  echo "  $label: read/write ok   ($(df -h --output=target "$path" 2>/dev/null | tail -1 | xargs))"
  return 0
}

check_disk_space() {
  echo "--- cache locations ---"
  # Printed in full because the failure mode is a variable quietly pointing
  # somewhere else, which is invisible unless the resolved paths are shown.
  for name in HF_HOME HF_HUB_CACHE HF_XET_CACHE TRANSFORMERS_CACHE \
              HF_DATASETS_CACHE TORCH_HOME TRITON_CACHE_DIR \
              TORCHINDUCTOR_CACHE_DIR VLLM_CACHE_ROOT TMPDIR; do
    local value="${!name:-}"
    local warn=""
    [[ "$value" == "$HOME"* ]] && warn="   <-- ON HOME"
    if [[ -n "$value" && "$value" != "$CACHE_BASE"* && "$value" != "$HOME"* ]]; then
      [[ -n "$LOCAL_RUNTIME_ROOT" && "$value" == "$LOCAL_RUNTIME_ROOT"* ]] \
        || warn="   <-- OUTSIDE CACHE_BASE"
    fi
    printf "  %-24s %s%s\n" "$name" "${value:-(unset)}" "$warn"
  done
  echo

  echo "--- filesystem health ---"
  local failed=0
  check_writable "$HF_HOME" "model cache" || failed=1
  check_writable "$TMPDIR" "tmp / compile" || failed=1
  echo

  echo "--- disk space ---"
  # Once the weights are cached the download headroom is no longer needed, and
  # demanding it would block a run that can actually proceed. orange sat at 100%
  # with 144 GiB free after the 57 GiB download, so the threshold has to reflect
  # what is still to be fetched rather than the model size.
  local cached_gib=0
  if [[ -d "$HF_HOME/hub" ]]; then
    cached_gib=$(du -sBG "$HF_HOME/hub" 2>/dev/null | tr -dc '0-9' || echo 0)
  fi
  local need=120
  if [[ ${cached_gib:-0} -ge 50 ]]; then
    need=20
    echo "  model already cached (~${cached_gib} GiB), so only working room is needed"
  fi
  check_space "$HF_HOME" "$need" "model cache" || failed=1
  check_space "$TMPDIR" 20 "tmp / compile" || failed=1
  check_space "$HOME" 5 "home (python, misc)" || true

  # df reports the filesystem, which on a cluster is usually far larger than the
  # per-user quota, so a home directory can look roomy and still be full. The
  # previous run failed exactly this way: orange had 200+ GiB, but the Xet staging
  # directory had fallen back to ~/.cache and home was already exhausted.
  echo "  quota check (per-user limits, not filesystem size):"
  local quota_out=""
  for cmd in "home_quota" "blue_quota" "quota -s"; do
    quota_out=$($cmd 2>/dev/null | head -8)
    [[ -n "$quota_out" ]] && { echo "$quota_out" | sed 's/^/    /'; break; }
  done
  [[ -z "$quota_out" ]] && echo "    (no quota tool found; check manually if a"\
    "download fails with 'Disk quota exceeded')"

  # Prove the caches really resolve off home, since these are read at import time
  # and a stale export would silently send tens of GiB to the wrong filesystem.
  echo "  resolved by huggingface_hub:"
  "$META_PY" - <<'PY' 2>/dev/null | sed 's/^/    /' || echo "    (could not query)"
import os
try:
    from huggingface_hub import constants as c
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        path = getattr(c, name, None)
        flag = "  <-- ON HOME" if path and path.startswith(os.path.expanduser("~")) else ""
        print(f"{name:<14} {path}{flag}")
except Exception as error:
    print(f"(huggingface_hub not importable: {error})")
PY
  if [[ $failed -ne 0 ]]; then
    echo
    echo "Not enough room for the model weights (~60 GiB, and Xet stages chunks"
    echo "before assembling them). The previous run died with 'Disk quota"
    echo "exceeded' partway through fetching them."
    echo
    echo "Re-run with the caches somewhere larger:"
    echo "    CACHE_BASE=/path/with/space bash run_experiments.sh $WHICH"
    echo
    echo "Note: an already-exported HF_HOME takes precedence over CACHE_BASE, so"
    echo "unset it first if it points somewhere small."
    echo "Reclaim space with:  rm -rf $HF_XET_CACHE   (staging only; safe)"
    exit 3
  fi
  echo
}

echo "=================================================================="
echo " PIRA measurements: $WHICH"
echo " started $(date -Is)   host $HOST"
echo " model   $MODEL   probe_layer $PROBE_LAYER   K $TOP_K (global)"
echo " results $OUT"
[[ -n "$LOCAL_RUNTIME_ROOT" ]] && echo " local runtime $LOCAL_RUNTIME_ROOT"
echo "=================================================================="
nvidia-smi 2>&1 | head -12 || echo "(nvidia-smi unavailable)"
echo
check_disk_space

if [[ "$WHICH" == "all" || "$WHICH" == "step2" ]]; then
  echo "--- Step 2 runtime preflight ---"
  echo "  python: $VLLM_PY"
  if ! command -v ninja >/dev/null 2>&1; then
    echo "  ninja: NOT FOUND (FlashInfer JIT cannot run)" >&2
    exit 4
  fi
  echo "  ninja: $(command -v ninja) ($(ninja --version 2>/dev/null || echo unknown))"
  echo
fi

# Started only now: the heartbeat reports compile-cache growth, and the cache
# variables above must be exported first or it would always print 0 MiB and look
# like nothing was happening.
heartbeat &
HEARTBEAT_PID=$!
# Kill the whole process group: killing the subshell alone leaves its `sleep`
# child running, which then outlives the script.
trap 'kill -- -$HEARTBEAT_PID 2>/dev/null || kill $HEARTBEAT_PID 2>/dev/null' \
  EXIT INT TERM

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
  # Checks that the installed transformers MoE layout is one this code knows how
  # to swap. A previous run downloaded 60 GiB, loaded the model, then skipped
  # every layer because the layout had changed -- this catches that in seconds.
  "$HF_PY" scripts/moe/pira_grouped_moe.py 2>&1 | tail -4
  LAYOUT_STATUS=$?
  if [[ $LAYOUT_STATUS -ne 0 ]]; then
    echo
    echo "The grouped-MoE layout self-check failed, so the grouped backends would"
    echo "match no layers on this transformers version. Fix that before running"
    echo "the comparison; the HF-only measurements below would still be valid."
  fi
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
  if [[ "$WHICH" == "all" || "$WHICH" == "step1" ]]; then
    "$HF_PY" -m pip list 2>/dev/null | grep -Ei "torch|transformers|grouped|megablocks|transformer-engine|triton" || true
  else
    echo "(not queried for a step2-only run)"
  fi
  echo
  echo "--- .venv-vllm ---"
  "$VLLM_PY" -m pip list 2>/dev/null | grep -Ei "^torch|vllm|transformers" || true
  echo
  echo "vLLM python: $VLLM_PY"
  echo "ninja: $(command -v ninja 2>/dev/null || echo not-found)"
  [[ -n "$LOCAL_RUNTIME_ROOT" ]] && echo "local runtime: $LOCAL_RUNTIME_ROOT"
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

GPU="$("$META_PY" -c "
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
