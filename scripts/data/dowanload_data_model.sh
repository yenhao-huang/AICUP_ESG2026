#!/usr/bin/env bash
# Download the shared data/model artifacts from Google Drive, extract archives,
# and install them into ./data and ./models.
#
# Run from anywhere:
#   bash scripts/data/dowanload_data_model.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DRIVE_FOLDER_URL="${DRIVE_FOLDER_URL:-https://drive.google.com/drive/u/0/folders/16rutragKIhpiCORINme9-vNvRR8Z3nOt}"
PYTHON="${PYTHON:-python3}"

if [ -e data ]; then
  echo "[error] refusing to continue: data already exists at $ROOT/data" >&2
  exit 1
fi

if [ -e models ]; then
  echo "[error] refusing to continue: models already exists at $ROOT/models" >&2
  exit 1
fi

command -v gdown >/dev/null 2>&1 || {
  echo "[error] missing dependency: gdown" >&2
  echo "        install with: pip install -U gdown" >&2
  exit 1
}

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "[error] missing dependency: $PYTHON" >&2
  exit 1
}

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/aicup_esg2026_artifacts.XXXXXX")"
DOWNLOAD_DIR="$WORKDIR/download"
EXTRACT_DIR="$WORKDIR/extracted"
mkdir -p "$DOWNLOAD_DIR" "$EXTRACT_DIR"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "### downloading artifacts"
echo "url=$DRIVE_FOLDER_URL"
echo "workdir=$WORKDIR"

gdown --folder "$DRIVE_FOLDER_URL" --output "$DOWNLOAD_DIR"

echo "### extracting archives"
"$PYTHON" - "$DOWNLOAD_DIR" "$EXTRACT_DIR" <<'PY'
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

download_dir = Path(sys.argv[1])
extract_dir = Path(sys.argv[2])

archive_count = 0
for path in download_dir.rglob("*"):
    if not path.is_file():
        continue

    lower_name = path.name.lower()
    if zipfile.is_zipfile(path):
        archive_count += 1
        target = extract_dir / path.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target)
    elif tarfile.is_tarfile(path):
        archive_count += 1
        target_name = lower_name
        for suffix in (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar"):
            if target_name.endswith(suffix):
                target_name = path.name[: -len(suffix)]
                break
        target = extract_dir / target_name
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as tf:
            tf.extractall(target)

if archive_count == 0:
    shutil.copytree(download_dir, extract_dir / "download", dirs_exist_ok=True)

print(f"extracted_archives={archive_count}")
PY

find_first_dir() {
  local name="$1"
  find "$EXTRACT_DIR" "$DOWNLOAD_DIR" -type d -name "$name" -print -quit
}

DATA_SRC="$(find_first_dir data || true)"
MODELS_SRC="$(find_first_dir models || true)"
MODELS_SUBMISSION_SRC="$(find_first_dir models_submission || true)"

if [ -z "$DATA_SRC" ]; then
  echo "[error] downloaded artifacts do not contain a data directory" >&2
  exit 1
fi

if [ -n "$MODELS_SRC" ]; then
  echo "### installing artifacts"
  mv "$DATA_SRC" data
  mv "$MODELS_SRC" models
elif [ -n "$MODELS_SUBMISSION_SRC" ]; then
  echo "### installing artifacts"
  mkdir -p models
  mv "$DATA_SRC" data
  mkdir -p models/submission
  shopt -s dotglob nullglob
  mv "$MODELS_SUBMISSION_SRC"/* models/submission/
else
  echo "[error] downloaded artifacts do not contain a models or models_submission directory" >&2
  exit 1
fi

echo "### artifact restore DONE"
echo "data -> $ROOT/data"
echo "models -> $ROOT/models"
