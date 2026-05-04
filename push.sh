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

# Stage modifications to tracked files (-u) plus paper content directories.
# .gitignore filters generated junk (.aux/.log/.out/etc.) and reference-only
# baselines (SteerMoE/, R2-Router/), so it is safe to add the content dirs
# wholesale; this captures new .tex files like sections/0_*.tex without
# manual `git add`.
git add -u
git add sections/ tables/ figures/ 2>/dev/null || true
for f in *.tex *.bib; do
    [ -f "$f" ] && git add "$f"
done

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
