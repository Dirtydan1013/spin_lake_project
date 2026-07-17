#!/bin/bash
# ─── Back up the gitignored local docs to origin/docs-backup ─────────────────
#
# docs/ (design docs, specs, experiment logs) and CLAUDE.md are deliberately
# gitignored — they are local research notes.  That also means a lost home
# directory loses them.  This script snapshots them onto a dedicated
# `docs-backup` branch in the SAME private repo, without touching the working
# tree, the index, or the currently checked-out branch.
#
# Usage:  ./scripts/backup_docs.sh        (run after meaningful doc updates)
# Restore: git show origin/docs-backup:docs/INDEX.md            (single file)
#          git checkout origin/docs-backup -- docs              (whole tree)
#
# Idempotent: if nothing changed since the last backup, no commit is made.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d docs ] || { echo "ERROR: no docs/ directory here" >&2; exit 1; }

# Stage docs/ + CLAUDE.md into a TEMPORARY index (repo index untouched).
# -u: path only — git must create the file itself (an existing empty file
# is rejected as a corrupt index).
TMPIDX=$(mktemp -u)
trap 'rm -f "$TMPIDX"' EXIT
GIT_INDEX_FILE="$TMPIDX" git add -f docs CLAUDE.md
TREE=$(GIT_INDEX_FILE="$TMPIDX" git write-tree)

# Parent = previous backup (local ref first, else the remote's).
PARENT_REF=""
if git rev-parse --verify -q refs/heads/docs-backup >/dev/null; then
    PARENT_REF="refs/heads/docs-backup"
elif git rev-parse --verify -q refs/remotes/origin/docs-backup >/dev/null; then
    PARENT_REF="refs/remotes/origin/docs-backup"
fi

if [ -n "$PARENT_REF" ]; then
    if [ "$(git rev-parse "$PARENT_REF^{tree}")" = "$TREE" ]; then
        echo "docs unchanged since last backup ($PARENT_REF) — nothing to do"
        exit 0
    fi
    COMMIT=$(git commit-tree "$TREE" -p "$PARENT_REF" \
             -m "docs backup $(date --iso-8601=seconds)")
else
    COMMIT=$(git commit-tree "$TREE" \
             -m "docs backup $(date --iso-8601=seconds)")
fi

git update-ref refs/heads/docs-backup "$COMMIT"
git push origin docs-backup
echo "docs backed up → origin/docs-backup ($COMMIT)"
