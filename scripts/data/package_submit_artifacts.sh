#!/usr/bin/env bash
# Build a zip containing the data/model artifacts required by scripts/submit.sh.
#
# Run from anywhere:
#   bash scripts/data/package_submit_artifacts.sh
#
# Optional upload:
#   GCS_URI=gs://your-bucket/path/ bash scripts/data/package_submit_artifacts.sh
#   DRIVE_FOLDER_URL=https://drive.google.com/drive/folders/<folder-id> bash scripts/data/package_submit_artifacts.sh
#   DRIVE_UPLOAD_MODE=split DRIVE_FOLDER_URL=https://drive.google.com/drive/folders/<folder-id> bash scripts/data/package_submit_artifacts.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-results/artifacts/submit_artifacts/$RUN_ID}"
ZIP_NAME="${ZIP_NAME:-aicup_esg2026_submit_artifacts.zip}"
ZIP_PATH="$OUT_ROOT/$ZIP_NAME"
MANIFEST="$OUT_ROOT/manifest.txt"
MANIFEST_DETAILS="$OUT_ROOT/manifest_details.tsv"
PYTHON="${PYTHON:-python3}"
GCS_URI="${GCS_URI:-}"
DRIVE_FOLDER_URL="${DRIVE_FOLDER_URL:-}"
DRIVE_FOLDER_ID="${DRIVE_FOLDER_ID:-}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:}"
ZIP_COMPRESSION="${ZIP_COMPRESSION:-store}"
DRIVE_UPLOAD_MODE="${DRIVE_UPLOAD_MODE:-single}"
DRIVE_SPLIT_SIZE="${DRIVE_SPLIT_SIZE:-3900M}"
PARTS_MANIFEST="$OUT_ROOT/${ZIP_NAME}.parts.txt"

ARTIFACT_PATHS=(
  "data/raw_data/vpesg4k_test_2000.json"
  "configs/prompts/stage4/boundary_rules_v4.txt"
  "models/submission/stage1"
  "models/submission/stage2"
  "models/submission/stage3"
  "models/submission/st12_fallback/gemma4_st12_mix/adapter_config.json"
  "models/submission/st12_fallback/gemma4_st12_mix/adapter_model.safetensors"
  "models/submission/st12_fallback/gemma4_st12_mix/README.md"
  "models/submission/st12_fallback/gemma4_st12_mix/tokenizer.json"
  "models/submission/st12_fallback/gemma4_st12_mix/tokenizer_config.json"
  "models/submission/st12_fallback/gemma4_st12_mix/train_meta.json"
  "models/submission/st12_fallback/gemma4_st12_mix/training_args.bin"
  "models/gemma/base/unsloth-gemma-4-12b"
)

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "[error] missing dependency: $PYTHON" >&2
  exit 1
}

if [ -n "$DRIVE_FOLDER_URL" ] && [ -z "$DRIVE_FOLDER_ID" ]; then
  DRIVE_FOLDER_ID="$("$PYTHON" - "$DRIVE_FOLDER_URL" <<'PY'
import re
import sys

url = sys.argv[1]
match = re.search(r"/folders/([A-Za-z0-9_-]+)", url)
if not match:
    match = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
if not match:
    raise SystemExit(f"cannot parse Google Drive folder id from: {url}")
print(match.group(1))
PY
)"
fi

require_path() {
  local path="$1"
  if [ ! -e "$path" ]; then
    echo "[error] missing required artifact path: $path" >&2
    exit 1
  fi
}

require_count() {
  local description="$1"
  local expected="$2"
  shift 2
  local count
  count="$(find -L "$@" -type f | wc -l)"
  if [ "$count" -lt "$expected" ]; then
    echo "[error] missing required artifact files for $description: found $count, expected at least $expected" >&2
    exit 1
  fi
}

for path in "${ARTIFACT_PATHS[@]}"; do
  require_path "$path"
done

require_count "Stage 1 checkpoints" 5 models/submission/stage1 -name best_st1.pt
require_count "Stage 2 checkpoints" 5 models/submission/stage2 -name best_st2.pt
require_count "Stage 3 checkpoint" 1 models/submission/stage3 -name best_multitask_st3.pt

case "$DRIVE_UPLOAD_MODE" in
  single|split) ;;
  *)
    echo "[error] DRIVE_UPLOAD_MODE must be one of: single, split (got: $DRIVE_UPLOAD_MODE)" >&2
    exit 1
    ;;
esac

rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

printf '%s\n' "${ARTIFACT_PATHS[@]}" > "$MANIFEST"

{
  printf 'path\ttype\tsize_bytes\n'
  for path in "${ARTIFACT_PATHS[@]}"; do
    if [ -d "$path" ]; then
      size="$(du -sbL "$path" | awk '{print $1}')"
      printf '%s\tdirectory\t%s\n' "$path" "$size"
    else
      size="$(stat -Lc '%s' "$path")"
      printf '%s\tfile\t%s\n' "$path" "$size"
    fi
  done
} > "$MANIFEST_DETAILS"

echo "### packaging submit artifacts"
echo "out_root=$OUT_ROOT"
echo "manifest=$MANIFEST"
echo "zip=$ZIP_PATH"

echo "### creating zip"
"$PYTHON" - "$ZIP_PATH" "$ZIP_COMPRESSION" "$MANIFEST" "$MANIFEST_DETAILS" "${ARTIFACT_PATHS[@]}" <<'PY'
import sys
import zipfile
import os
from pathlib import Path

zip_path = Path(sys.argv[1])
compression_name = sys.argv[2]
manifest = Path(sys.argv[3])
manifest_details = Path(sys.argv[4])
artifact_paths = [Path(arg) for arg in sys.argv[5:]]

if compression_name == "store":
    compression = zipfile.ZIP_STORED
    kwargs = {}
elif compression_name == "deflate":
    compression = zipfile.ZIP_DEFLATED
    kwargs = {"compresslevel": 6}
else:
    raise SystemExit(f"unsupported ZIP_COMPRESSION={compression_name}; use store or deflate")

def add_file(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    zf.write(path, arcname)

with zipfile.ZipFile(zip_path, "w", compression=compression, allowZip64=True, **kwargs) as zf:
    zf.write(manifest, "artifact_manifest.txt")
    zf.write(manifest_details, "artifact_manifest_details.tsv")

    for artifact in artifact_paths:
        if artifact.is_dir():
            for root, _, files in os.walk(artifact, followlinks=True):
                for name in sorted(files):
                    path = Path(root) / name
                    zf.write(path.resolve(), path.as_posix())
        elif artifact.is_file():
            zf.write(artifact.resolve(), artifact.as_posix())
        else:
            raise SystemExit(f"unsupported artifact path: {artifact}")
PY

du -sh "$ZIP_PATH"

"$PYTHON" - "$ZIP_PATH" "$DRIVE_SPLIT_SIZE" "$PARTS_MANIFEST" <<'PY'
import math
import re
import sys
from pathlib import Path

zip_path = Path(sys.argv[1])
split_size = sys.argv[2]
parts_manifest = Path(sys.argv[3])

match = re.fullmatch(r"(\d+)([KMGTP]?)", split_size)
if not match:
    raise SystemExit(f"unsupported DRIVE_SPLIT_SIZE={split_size}; use forms like 3900M or 4G")

value = int(match.group(1))
unit = match.group(2)
multiplier = {
    "": 1,
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
    "P": 1024**5,
}[unit]
part_bytes = value * multiplier
total_bytes = zip_path.stat().st_size
part_count = math.ceil(total_bytes / part_bytes)

with parts_manifest.open("w", encoding="utf-8") as f:
    f.write(f"zip_name\t{zip_path.name}\n")
    f.write(f"zip_size_bytes\t{total_bytes}\n")
    f.write(f"split_size\t{split_size}\n")
    f.write(f"part_count\t{part_count}\n")
    f.write("parts\n")
    for index in range(part_count):
        f.write(f"{zip_path.name}.part-{index:03d}\n")
PY

upload_drive_single() {
  local path="$1"
  rclone copy "$path" "$RCLONE_REMOTE" --drive-root-folder-id "$DRIVE_FOLDER_ID" --progress
}

upload_drive_split_zip() {
  local path="$1"
  local name
  name="$(basename "$path")"

  command -v split >/dev/null 2>&1 || {
    echo "[error] DRIVE_UPLOAD_MODE=split requires GNU split" >&2
    exit 1
  }

  export RCLONE_REMOTE DRIVE_FOLDER_ID
  echo "### streaming split upload to Google Drive"
  echo "zip=$path"
  echo "split_size=$DRIVE_SPLIT_SIZE"
  echo "parts_manifest=$PARTS_MANIFEST"

  split \
    --bytes="$DRIVE_SPLIT_SIZE" \
    --numeric-suffixes=0 \
    --suffix-length=3 \
    --filter='part_name=$(basename "$FILE"); rclone rcat "${RCLONE_REMOTE}${part_name}" --drive-root-folder-id "$DRIVE_FOLDER_ID" --progress' \
    "$path" \
    "${name}.part-"
}

if [ -n "$GCS_URI" ]; then
  command -v gcloud >/dev/null 2>&1 || {
    echo "[error] GCS_URI was set, but gcloud is not installed or not on PATH" >&2
    exit 1
  }

  echo "### uploading to Google Cloud Storage"
  echo "destination=$GCS_URI"
  gcloud storage cp "$ZIP_PATH" "$GCS_URI"
  gcloud storage cp "$MANIFEST" "$GCS_URI"
  gcloud storage cp "$MANIFEST_DETAILS" "$GCS_URI"
fi

if [ -n "$DRIVE_FOLDER_ID" ]; then
  command -v rclone >/dev/null 2>&1 || {
    echo "[error] DRIVE_FOLDER_URL/DRIVE_FOLDER_ID was set, but rclone is not installed or not on PATH" >&2
    exit 1
  }

  echo "### uploading to Google Drive"
  echo "folder_id=$DRIVE_FOLDER_ID"
  echo "remote=$RCLONE_REMOTE"
  echo "mode=$DRIVE_UPLOAD_MODE"
  case "$DRIVE_UPLOAD_MODE" in
    single)
      upload_drive_single "$ZIP_PATH"
      ;;
    split)
      upload_drive_split_zip "$ZIP_PATH"
      upload_drive_single "$PARTS_MANIFEST"
      ;;
  esac
  upload_drive_single "$MANIFEST"
  upload_drive_single "$MANIFEST_DETAILS"
fi

echo "### submit artifact package DONE"
echo "zip=$ZIP_PATH"
echo "manifest=$MANIFEST"
echo "manifest_details=$MANIFEST_DETAILS"
