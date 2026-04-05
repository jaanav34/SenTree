#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/sync_from_gautschi_wsl.sh <SRC> <DEST>

Examples:
  # With jump host (recommended on Gautschi)
  export SENTREE_SSH_JUMP="shah958@gautschi.rcac.purdue.edu"
  scripts/sync_from_gautschi_wsl.sh \
    "shah958@a242.gautschi.rcac.purdue.edu:/scratch/gautschi/shah958/sentree/SenTree/" \
    "/mnt/c/Users/fireb/Downloads/Projects/SenTree/"

Notes:
  - Uses scripts/rsync_excludes.txt to avoid syncing venv/node_modules/.env/etc.
  - Set SENTREE_RSYNC_DELETE=1 to mirror deletions from SRC to DEST.
  - Set SENTREE_RSYNC_EXTRA="..." to append extra rsync flags.
USAGE
}

SRC="${1:-}"
DEST="${2:-}"

if [[ -z "$SRC" || -z "$DEST" ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXCLUDES_FILE="${SENTREE_RSYNC_EXCLUDES_FILE:-$SCRIPT_DIR/rsync_excludes.txt}"

if [[ ! -f "$EXCLUDES_FILE" ]]; then
  echo "ERROR: excludes file not found: $EXCLUDES_FILE" >&2
  exit 2
fi

SSH_JUMP="${SENTREE_SSH_JUMP:-}"
SSH_OPTS=(
  -o ServerAliveInterval=60
  -o ServerAliveCountMax=3
)

if [[ -n "$SSH_JUMP" ]]; then
  RSH=(ssh -J "$SSH_JUMP" "${SSH_OPTS[@]}")
else
  RSH=(ssh "${SSH_OPTS[@]}")
fi

RSYNC_ARGS=(
  -az
  --partial
  --human-readable
  --info=progress2
  --exclude-from "$EXCLUDES_FILE"
  -e "${RSH[*]}"
)

if [[ "${SENTREE_RSYNC_DELETE:-0}" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ -n "${SENTREE_RSYNC_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  RSYNC_ARGS+=(${SENTREE_RSYNC_EXTRA})
fi

echo "rsync ${RSYNC_ARGS[*]} \"$SRC\" \"$DEST\""
rsync "${RSYNC_ARGS[@]}" "$SRC" "$DEST"
