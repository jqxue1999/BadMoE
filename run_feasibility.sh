#!/usr/bin/env bash
# One-click grouped-GEMM feasibility probe: run, log, commit, push.
#
# Answers whether adapting the PIRA probe to a grouped-GEMM MoE backend is worth
# doing, before any adapter code is written. See
# scripts/moe/probe_grouped_gemm_feasibility.py for what is measured and why.
#
#   bash run_feasibility.sh
#
# Everything lands in results/feasibility/<timestamp>-<host>/ and is pushed to a
# branch, so the log can be reviewed without copy-pasting terminal output.
#
# Environment overrides:
#   PYTHON_BIN     interpreter to use (default: .venv/bin/python if present)
#   SEQ_LENS       prompt lengths to probe (default: "128 512 2048")
#   BRANCH         branch to push to (default: current branch)
#   NO_PUSH=1      run and commit locally, skip the push
#   NO_COMMIT=1    run only, leave results uncommitted

set -uo pipefail   # NOT -e: a failing probe must still be logged and pushed

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d-%H%M%S)"
HOST="$(hostname -s 2>/dev/null || echo unknown)"
RESULT_DIR="results/feasibility/${STAMP}-${HOST}"
mkdir -p "$RESULT_DIR"

LOG="$RESULT_DIR/run.log"
JSON="$RESULT_DIR/feasibility.json"
ENVFILE="$RESULT_DIR/environment.txt"

# Everything below is teed into the log as well as shown live.
exec > >(tee -a "$LOG") 2>&1

echo "=================================================================="
echo " PIRA grouped-GEMM feasibility probe"
echo " started   $(date -Is)"
echo " host      $HOST"
echo " results   $RESULT_DIR"
echo "=================================================================="
echo

# ---- interpreter -------------------------------------------------------------
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
echo "interpreter: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('python', sys.version.split()[0])" || {
  echo "FATAL: $PYTHON_BIN is not usable. Run setup_env.sh first."
  exit 2
}
echo

# ---- environment capture -----------------------------------------------------
# Recorded because the headline risk is hardware/toolchain specific: the
# grouped_gemm CUDA source hardcodes ::cutlass::arch::Sm80 (Ampere) while B200 is
# sm_100, so knowing the exact GPU, CUDA and package versions is part of the result.
{
  echo "date: $(date -Is)"
  echo "host: $HOST"
  echo "pwd:  $REPO_ROOT"
  echo "git:  $(git rev-parse --short HEAD 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo
  echo "--- nvidia-smi ---"
  nvidia-smi 2>&1 | head -15 || echo "(nvidia-smi unavailable)"
  echo
  echo "--- python packages ---"
  "$PYTHON_BIN" - <<'PY' 2>&1
import importlib.metadata as md
for pkg in ("torch", "grouped_gemm", "megablocks", "transformer_engine",
            "transformers", "vllm", "triton"):
    try:
        print(f"{pkg:<20} {md.version(pkg)}")
    except Exception:
        print(f"{pkg:<20} (not installed)")
PY
  echo
  echo "--- torch / CUDA ---"
  "$PYTHON_BIN" - <<'PY' 2>&1
import torch
print("torch          ", torch.__version__)
print("cuda runtime   ", torch.version.cuda)
print("cuda available ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         ", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("capability      sm_%d%d" % cap)
    print("memory          %.1f GiB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
PY
} | tee "$ENVFILE"
echo

# ---- import checks -----------------------------------------------------------
# Reported rather than fatal: "grouped_gemm will not build on Blackwell" is
# itself one of the answers this probe exists to produce.
echo "--- backend import check ---"
"$PYTHON_BIN" - <<'PY'
for name, module in (("grouped_gemm", "grouped_gemm"),
                     ("MegaBlocks", "megablocks"),
                     ("TransformerEngine", "transformer_engine.pytorch")):
    try:
        __import__(module)
        print(f"  {name:<20} OK")
    except Exception as e:
        print(f"  {name:<20} unavailable -> {type(e).__name__}: {e}")
PY
echo

# ---- correctness gate --------------------------------------------------------
# Cheap CPU check that the grouped sort/scatter reconstruction is right. A bug
# there would produce plausible timings for the wrong computation, so it runs
# before anything is measured.
echo "--- correctness self-check (CPU) ---"
"$PYTHON_BIN" scripts/moe/probe_grouped_gemm_feasibility.py --self-check
SELF_CHECK_STATUS=$?
if [[ $SELF_CHECK_STATUS -ne 0 ]]; then
  echo
  echo "SELF-CHECK FAILED -- the timing numbers would describe the wrong"
  echo "computation, so the probe is being skipped. Log saved to $LOG."
  git add -f "$RESULT_DIR" >/dev/null 2>&1
  exit $SELF_CHECK_STATUS
fi
echo

# ---- the probe ---------------------------------------------------------------
SEQ_LENS="${SEQ_LENS:-128 512 2048}"
echo "--- running probe (seq lens: $SEQ_LENS) ---"
echo
"$PYTHON_BIN" scripts/moe/probe_grouped_gemm_feasibility.py \
  --seq-lens $SEQ_LENS \
  --output "$JSON"
PROBE_STATUS=$?
echo
if [[ $PROBE_STATUS -eq 0 ]]; then
  echo "probe exit status: 0 (ok)"
else
  echo "probe exit status: $PROBE_STATUS (see log above; partial results still recorded)"
fi
echo

# ---- commit and push ---------------------------------------------------------
if [[ "${NO_COMMIT:-0}" == "1" ]]; then
  echo "NO_COMMIT=1, leaving results in $RESULT_DIR"
  exit $PROBE_STATUS
fi

BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null)}"
echo "--- committing results to $BRANCH ---"

# results/ is force-added because the repo's .gitignore excludes result dirs by
# default; these files are small text and are the point of the run.
git add -f "$RESULT_DIR" 2>&1

if git diff --cached --quiet 2>/dev/null; then
  echo "nothing to commit (no result files produced?)"
  exit $PROBE_STATUS
fi

GPU_NAME="$("$PYTHON_BIN" -c "
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')
" 2>/dev/null || echo unknown)"

git -c user.name="${GIT_NAME:-$(git config user.name || echo 'PIRA bench')}" \
    -c user.email="${GIT_EMAIL:-$(git config user.email || echo 'bench@localhost')}" \
    commit -q -m "Feasibility probe results: ${STAMP} on ${GPU_NAME}

Grouped-GEMM MoE backend feasibility at Qwen3-30B-A3B shapes.
Host ${HOST}, probe exit status ${PROBE_STATUS}.
Sequence lengths: ${SEQ_LENS}

Full log and environment capture in ${RESULT_DIR}." 2>&1

echo "committed $(git rev-parse --short HEAD)"
echo

if [[ "${NO_PUSH:-0}" == "1" ]]; then
  echo "NO_PUSH=1, not pushing. To push later:  git push origin $BRANCH"
  exit $PROBE_STATUS
fi

echo "--- pushing to origin/$BRANCH ---"
if git push origin "$BRANCH" 2>&1; then
  echo
  echo "=================================================================="
  echo " DONE. Results pushed."
  echo " Log:     $RESULT_DIR/run.log"
  echo " JSON:    $RESULT_DIR/feasibility.json"
  echo " Commit:  $(git rev-parse --short HEAD) on $BRANCH"
  echo "=================================================================="
else
  echo
  echo "push failed (results are committed locally)."
  echo "retry with:  git push origin $BRANCH"
fi

exit $PROBE_STATUS
