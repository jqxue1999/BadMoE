#!/usr/bin/env bash
# Reuse a restored FlashInfer shared object whose package/GPU cache tag was
# already validated by run_experiments.sh. Delegate every other invocation to
# the environment's real Ninja executable.
set -uo pipefail

real_ninja="${BADMOE_REAL_NINJA:?BADMOE_REAL_NINJA is not set}"
build_dir=""
build_file=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    -C) ((i++)); build_dir="${args[$i]:-}" ;;
    -f) ((i++)); build_file="${args[$i]:-}" ;;
  esac
done

flashinfer_root="${FLASHINFER_WORKSPACE_BASE%/}/.cache/flashinfer/"
if [[ -n "${BADMOE_RESTORED_CACHE_TAG:-}" &&
      "$build_dir/" == "$flashinfer_root"* ]]; then
  [[ -n "$build_file" ]] || build_file="$build_dir/build.ninja"
  if [[ -f "$build_file" ]]; then
    output=$(sed -n 's/^default //p' "$build_file" | tail -1)
    if [[ -n "$output" && -f "$output" ]]; then
      echo "ninja: reusing restored FlashInfer binary $output"
      exit 0
    fi
  fi
fi

exec "$real_ninja" "$@"
