#!/usr/bin/env bash
# Push paper edits to Overleaf (origin) and GitHub (github, if present).
#
# Usage:
#   ./push.sh                       # auto-message "update paper"
#   ./push.sh "your commit message" # custom message
#
# What it does:
#   1. cd into paper/ (the script's own directory)
#   2. Stage tracked-file modifications + the figures/ folder
#   3. Commit if there are staged changes (skip if nothing changed)
#   4. Push to origin/master (Overleaf)
#   5. Also push to github/master if that remote exists

set -euo pipefail

cd "$(dirname "$0")"

MSG="${1:-update paper}"

# Stage modifications to tracked files + any new figures.
# (`-u` skips new untracked files like core dumps, slurm logs, etc.)
git add -u
git add figures/ 2>/dev/null || true

if git diff --cached --quiet; then
    echo "Nothing to commit. Working tree clean relative to HEAD."
    exit 0
fi

echo "===> Committing: \"$MSG\""
git commit -m "$MSG"

echo "===> Pushing to Overleaf (origin)..."
git push origin master

if git remote | grep -qx github; then
    echo "===> Pushing to GitHub (github)..."
    git push github master
fi

echo "Done."
