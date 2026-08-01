#!/usr/bin/env bash
# Set up the environment for the PIRA benchmarks with uv.
#
#   bash setup_env.sh              # everything: both envs, every backend
#   bash setup_env.sh --no-vllm    # probe env only; skips the slow vLLM install
#   bash setup_env.sh --no-push    # log locally, do not push
#
# The default installs everything needed for the whole benchmark suite in one
# pass, so there is no second setup round later. The full run is logged to
# results/setup/<timestamp>-<host>/ and pushed, so install failures can be
# reviewed without pasting terminal output.
#
# Two virtualenvs, because vLLM pins its own Transformers stack and the probe
# needs a version it can hook:
#
#   .venv        torch + transformers        probe, equivalence tests, feasibility
#   .venv-vllm   vllm                        Original and PIRA-generation arms
#
# The grouped-GEMM backends are OPTIONAL and installed best-effort. Failure is
# expected and informative: tgale96/grouped_gemm hardcodes
# ::cutlass::arch::Sm80 (Ampere) in its CUDA source, so on Blackwell (sm_100) it
# may not build at all. The feasibility probe reports whichever backends imported
# and skips the rest, so a partial install still produces a usable answer.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# vLLM is installed by default: the serving arms (Original and PIRA generation)
# need it, and provisioning both environments in one pass avoids a second setup
# round later. --no-vllm skips it when only the feasibility probe is wanted.
WANT_VLLM=1
NO_PUSH="${NO_PUSH:-0}"
for arg in "$@"; do
  case "$arg" in
    --no-vllm) WANT_VLLM=0 ;;
    --no-push) NO_PUSH=1 ;;
    --vllm|--all) ;;   # accepted for compatibility; both are now the default
    -h|--help)
      sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# ---- logging ----------------------------------------------------------------
# The whole run is logged and pushed, because a FAILED backend build is itself a
# result: grouped_gemm hardcodes ::cutlass::arch::Sm80, so whether it builds on
# Blackwell -- and the exact compiler error when it does not -- is what decides
# whether the grouped-GEMM route is viable at all.
STAMP="$(date +%Y%m%d-%H%M%S)"
HOST="$(hostname -s 2>/dev/null || echo unknown)"
SETUP_DIR="results/setup/${STAMP}-${HOST}"
mkdir -p "$SETUP_DIR"
SETUP_LOG="$SETUP_DIR/setup.log"
exec > >(tee -a "$SETUP_LOG") 2>&1

echo "=================================================================="
echo " PIRA environment setup"
echo " started  $(date -Is)"
echo " host     $HOST"
echo " log      $SETUP_LOG"
echo "=================================================================="

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-}"     # empty = let uv pick a build for this CUDA

# ---- keep caches off the home filesystem ------------------------------------
# uv, pip, HF and the CUDA compilers all default to $HOME regardless of where the
# project lives, so running from scratch storage still fills a small home quota
# ("No space left on device" while writing ~/.cache/uv). Everything is therefore
# redirected next to the repository by default. Putting the cache on the same
# filesystem as .venv also restores hardlinking, which makes installs much faster
# than the copy fallback.
CACHE_BASE="${CACHE_BASE:-$REPO_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_BASE/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_BASE/pip}"
export HF_HOME="${HF_HOME:-$CACHE_BASE/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_BASE/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_BASE/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_BASE/inductor}"
# grouped_gemm compiles CUTLASS from source; its scratch files are large and would
# otherwise land in /tmp or $HOME.
export TMPDIR="${TMPDIR_OVERRIDE:-$CACHE_BASE/tmp}"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
# Same filesystem as the venv now, so hardlinks work; copy remains the fallback.
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"

step() { echo; echo "=== $* ==="; }

# ---- disk space ---------------------------------------------------------------
# The full install is roughly 15-20 GiB (torch + CUDA libraries in two venvs, plus
# the wheel cache). Checked up front because running out mid-install leaves a
# half-populated venv that then fails in confusing ways.
step "disk space"
echo "caches and venvs will be written under: $CACHE_BASE"
AVAIL=$(df -BG --output=avail "$REPO_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
MOUNT=$(df -h --output=target "$REPO_ROOT" 2>/dev/null | tail -1 | xargs)
HOME_AVAIL=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "  repo   $REPO_ROOT"
echo "         mount ${MOUNT:-unknown}, ${AVAIL:-?} GiB free"
echo "  home   $HOME"
echo "         ${HOME_AVAIL:-?} GiB free  (not used for caches; see above)"
if [[ -n "$AVAIL" && "$AVAIL" -lt 25 ]]; then
  echo
  echo "WARNING: only ${AVAIL} GiB free where the venvs will go. The full install"
  echo "         needs roughly 15-20 GiB. Point CACHE_BASE at somewhere larger,"
  echo "         e.g.  CACHE_BASE=/scratch/\$USER/.cache bash setup_env.sh"
fi

# ---- uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  step "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh || {
    echo "FATAL: could not install uv. Install it manually: https://docs.astral.sh/uv/"
    exit 2
  }
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "uv: $(command -v uv) ($(uv --version 2>/dev/null))"

# ---- probe env --------------------------------------------------------------
step "creating .venv (probe / feasibility), python $PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" .venv || exit 2
VPY="$REPO_ROOT/.venv/bin/python"

step "installing torch + transformers into .venv"
if [[ -n "$TORCH_VERSION" ]]; then
  uv pip install --python "$VPY" "torch==$TORCH_VERSION" || exit 2
else
  uv pip install --python "$VPY" torch || exit 2
fi
uv pip install --python "$VPY" \
  transformers accelerate safetensors sentencepiece numpy packaging ninja || exit 2

"$VPY" - <<'PY'
import torch
print("torch          ", torch.__version__)
print("cuda runtime   ", torch.version.cuda)
print("cuda available ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         ", torch.cuda.get_device_name(0))
    print("capability      sm_%d%d" % torch.cuda.get_device_capability(0))
PY

# ---- grouped-GEMM backends (best effort) ------------------------------------
# Ordered by how likely they are to work on current hardware. TransformerEngine
# is NVIDIA-maintained and treats Blackwell as first class, so it is tried first
# and is the one most likely to be usable.
step "installing grouped-GEMM backends (optional, failures are expected)"

echo "--- TransformerEngine (Megatron-Core MoE path; NVIDIA-maintained) ---"
uv pip install --python "$VPY" --no-build-isolation transformer_engine[pytorch] \
  && echo "  transformer_engine: installed" \
  || echo "  transformer_engine: FAILED (probe will report it unavailable)"

echo
echo "--- grouped_gemm (MegaBlocks kernel; CUDA arch hardcoded to Sm80) ---"
# GROUPED_GEMM_CUTLASS=1 selects the real grouped kernel instead of the
# conservative cuBLAS loop, which is the whole point of using it.
GROUPED_GEMM_CUTLASS=1 uv pip install --python "$VPY" --no-build-isolation \
  git+https://github.com/tgale96/grouped_gemm.git \
  && echo "  grouped_gemm: installed" \
  || echo "  grouped_gemm: FAILED (expected on sm_90+; this is a real result)"

echo
echo "--- MegaBlocks (dMoE layers on top of grouped_gemm) ---"
uv pip install --python "$VPY" --no-build-isolation megablocks \
  && echo "  megablocks: installed" \
  || echo "  megablocks: FAILED"

# ---- vLLM env ---------------------------------------------------------------
if [[ $WANT_VLLM -eq 1 ]]; then
  step "creating .venv-vllm (serving arms: Original and PIRA generation)"
  echo "this pulls a large wheel and can take several minutes;"
  echo "skip it with --no-vllm if you only need the feasibility probe."
  uv venv --python "$PYTHON_VERSION" .venv-vllm || exit 2
  if uv pip install --python "$REPO_ROOT/.venv-vllm/bin/python" vllm; then
    echo "  vllm: installed"
    "$REPO_ROOT/.venv-vllm/bin/python" - <<'PY' 2>&1 || true
import vllm, torch
print("  vllm          ", vllm.__version__)
print("  torch (vllm)  ", torch.__version__)
PY
  else
    echo "  vllm: FAILED (the probe in .venv is unaffected)"
  fi
else
  echo
  echo "skipping vLLM (--no-vllm). The serving arms will need it later:"
  echo "    bash setup_env.sh          # re-run to add .venv-vllm"
fi

# ---- summary ----------------------------------------------------------------
step "final summary"
echo ".venv  (probe, equivalence tests, feasibility):"
"$VPY" - <<'PY'
for name, module in (("torch", "torch"),
                     ("transformers", "transformers"),
                     ("grouped_gemm", "grouped_gemm"),
                     ("megablocks", "megablocks"),
                     ("transformer_engine", "transformer_engine.pytorch")):
    try:
        __import__(module)
        print(f"  {name:<20} OK")
    except Exception as e:
        print(f"  {name:<20} unavailable -> {type(e).__name__}")
PY

echo
echo ".venv-vllm  (Original and PIRA generation arms):"
if [[ -x "$REPO_ROOT/.venv-vllm/bin/python" ]]; then
  "$REPO_ROOT/.venv-vllm/bin/python" - <<'PY'
for name, module in (("torch", "torch"), ("vllm", "vllm")):
    try:
        __import__(module)
        print(f"  {name:<20} OK")
    except Exception as e:
        print(f"  {name:<20} unavailable -> {type(e).__name__}")
PY
else
  echo "  (not created; re-run without --no-vllm)"
fi

cat <<EOF

=== done ===

Next: run the feasibility probe (logs + JSON are committed and pushed for review)

    bash run_feasibility.sh

Sanity checks that need no GPU and take seconds:

    .venv/bin/python scripts/moe/test_probe_equivalence_cpu.py
    .venv/bin/python scripts/moe/pira_vllm_routing.py
    .venv/bin/python scripts/moe/rebuttal_vllm_pira_benchmark.py --check-workload

If a grouped-GEMM backend failed to install, run the probe anyway. It reports
which backends were importable and measures the rest; "does not build on this
GPU" is one of the answers we need.
EOF

# ---- capture the environment as a file, then commit and push -----------------
{
  echo "date: $(date -Is)"
  echo "host: $HOST"
  echo "git:  $(git rev-parse --short HEAD 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "uv:   $(uv --version 2>/dev/null)"
  echo
  echo "--- nvidia-smi ---"
  nvidia-smi 2>&1 | head -15 || echo "(unavailable)"
  echo
  echo "--- .venv packages ---"
  uv pip list --python "$VPY" 2>/dev/null || echo "(unavailable)"
  if [[ -x "$REPO_ROOT/.venv-vllm/bin/python" ]]; then
    echo
    echo "--- .venv-vllm packages ---"
    uv pip list --python "$REPO_ROOT/.venv-vllm/bin/python" 2>/dev/null || echo "(unavailable)"
  fi
} > "$SETUP_DIR/environment.txt" 2>&1

step "saving setup log"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
# Force-added: results/ is gitignored so stray local runs stay out of commits,
# but this log is the point of the run.
git add -f "$SETUP_DIR" >/dev/null 2>&1

if git diff --cached --quiet 2>/dev/null; then
  echo "nothing to commit"
else
  git -c user.name="${GIT_NAME:-$(git config user.name || echo 'PIRA bench')}" \
      -c user.email="${GIT_EMAIL:-$(git config user.email || echo 'bench@localhost')}" \
      commit -q -m "Environment setup log: ${STAMP} on ${HOST}

Package versions and backend availability from setup_env.sh, including any
grouped-GEMM build failures." >/dev/null 2>&1
  echo "committed $(git rev-parse --short HEAD)"

  if [[ "$NO_PUSH" == "1" ]]; then
    echo "--no-push set; push later with:  git push origin $BRANCH"
  elif git push origin "$BRANCH" >/dev/null 2>&1; then
    echo "pushed to origin/$BRANCH"
  else
    echo "push failed (log is committed locally; retry: git push origin $BRANCH)"
  fi
fi

echo
echo "setup log: $SETUP_LOG"
