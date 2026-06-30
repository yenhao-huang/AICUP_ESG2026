#!/usr/bin/env bash
# Download the shared data/model artifacts from Google Drive, rebuild split
# archive shards when present, extract archives, and install them into ./data
# and ./models.
#
# Run from anywhere:
#   bash scripts/data/dowanload_data_model.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DRIVE_FOLDER_URL="${DRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/16rutragKIhpiCORINme9-vNvRR8Z3nOt}"
PYTHON="${PYTHON:-python3}"
CACHE_ROOT="${CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/aicup_esg2026/download_data_model}"
REFRESH_CACHE="${REFRESH_CACHE:-0}"

if [ -e data ]; then
  echo "[error] refusing to continue: data already exists at $ROOT/data" >&2
  exit 1
fi

if [ -e models ]; then
  echo "[error] refusing to continue: models already exists at $ROOT/models" >&2
  exit 1
fi

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "[error] missing dependency: $PYTHON" >&2
  exit 1
}

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/aicup_esg2026_artifacts.XXXXXX")"
EXTRACT_DIR="$WORKDIR/extracted"
CACHE_TMP=""
mkdir -p "$EXTRACT_DIR"

cleanup() {
  rm -rf "$WORKDIR"
  if [ -n "$CACHE_TMP" ]; then
    rm -rf "$CACHE_TMP"
  fi
}
trap cleanup EXIT

CACHE_KEY="$("$PYTHON" - "$DRIVE_FOLDER_URL" <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest()[:16])
PY
)"
CACHE_ENTRY="$CACHE_ROOT/$CACHE_KEY"
CACHE_DOWNLOAD_DIR="$CACHE_ENTRY/download"
CACHE_MARKER="$CACHE_ENTRY/.complete"

echo "### downloading artifacts"
echo "url=$DRIVE_FOLDER_URL"
echo "cache=$CACHE_ENTRY"
echo "workdir=$WORKDIR"

if [ "$REFRESH_CACHE" != "1" ] && [ -f "$CACHE_MARKER" ] && [ -d "$CACHE_DOWNLOAD_DIR" ]; then
  echo "### using cached download"
  DOWNLOAD_DIR="$CACHE_DOWNLOAD_DIR"
else
  command -v gdown >/dev/null 2>&1 || {
    echo "[error] missing dependency: gdown" >&2
    echo "        install with: pip install -U gdown" >&2
    exit 1
  }

  mkdir -p "$CACHE_ROOT"
  CACHE_TMP="$(mktemp -d "$CACHE_ROOT/.tmp.${CACHE_KEY}.XXXXXX")"
  DOWNLOAD_DIR="$CACHE_TMP/download"
  mkdir -p "$DOWNLOAD_DIR"

  if ! gdown --folder "$DRIVE_FOLDER_URL" --output "$DOWNLOAD_DIR"; then
    cat >&2 <<'EOF'
[error] gdown failed to download the Google Drive artifacts.

Common causes:
  1. The Drive folder or one of its files is not shared as "Anyone with the link"
     with Viewer permission.
  2. Google Drive temporarily blocked the file because it has had too many
     accesses or downloads.
  3. The folder contains shortcuts or files whose inherited sharing permission
     does not allow public download.

Fix the Drive sharing settings, wait for the quota block to clear, or upload the
artifacts to a download host that supports large public files.
EOF
    exit 1
  fi

  rm -rf "$CACHE_ENTRY"
  mkdir -p "$CACHE_ENTRY"
  mv "$DOWNLOAD_DIR" "$CACHE_DOWNLOAD_DIR"
  touch "$CACHE_MARKER"
  DOWNLOAD_DIR="$CACHE_DOWNLOAD_DIR"
  rm -rf "$CACHE_TMP"
  CACHE_TMP=""
fi

echo "### extracting archives"
"$PYTHON" - "$DOWNLOAD_DIR" "$EXTRACT_DIR" "$WORKDIR" <<'PY'
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

download_dir = Path(sys.argv[1])
extract_dir = Path(sys.argv[2])
workdir = Path(sys.argv[3])


def parse_size(value: str) -> int:
    value = value.strip()
    units = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }
    suffix = value[-1:].upper()
    if suffix in units and not suffix.isdigit():
        return int(value[:-1]) * units[suffix]
    return int(value)


def parse_parts_manifest(path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    parts: list[str] = []
    in_parts = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "parts":
            in_parts = True
            continue
        if in_parts:
            parts.append(line)
            continue
        key, _, value = line.partition("\t")
        if key and value:
            metadata[key] = value

    metadata["parts"] = parts
    return metadata


def find_file_by_name(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if not matches:
        raise SystemExit(f"[error] missing shard listed in manifest: {name}")
    if len(matches) > 1:
        raise SystemExit(f"[error] shard name is ambiguous: {name}")
    return matches[0]


def rebuild_split_zip() -> Path | None:
    manifests = sorted(download_dir.rglob("*.zip.parts.txt"))
    if not manifests:
        return None
    if len(manifests) > 1:
        names = "\n".join(str(path) for path in manifests)
        raise SystemExit(f"[error] found multiple zip parts manifests:\n{names}")

    manifest_path = manifests[0]
    manifest = parse_parts_manifest(manifest_path)
    zip_name = str(manifest.get("zip_name", "")).strip()
    parts = manifest.get("parts", [])
    if not zip_name:
        raise SystemExit(f"[error] missing zip_name in {manifest_path}")
    if not isinstance(parts, list) or not parts:
        raise SystemExit(f"[error] missing parts list in {manifest_path}")

    expected_count = int(str(manifest.get("part_count", len(parts))))
    if expected_count != len(parts):
        raise SystemExit(
            f"[error] manifest part_count={expected_count}, but listed {len(parts)} parts"
        )

    expected_zip_size = int(str(manifest["zip_size_bytes"]))
    split_size = parse_size(str(manifest.get("split_size", "0")))
    shard_paths = [find_file_by_name(download_dir, name) for name in parts]

    actual_total = sum(path.stat().st_size for path in shard_paths)
    if actual_total != expected_zip_size:
        raise SystemExit(
            f"[error] shard sizes sum to {actual_total}, expected {expected_zip_size}"
        )

    if split_size:
        for index, path in enumerate(shard_paths):
            expected = (
                split_size
                if index < len(shard_paths) - 1
                else expected_zip_size - split_size * (len(shard_paths) - 1)
            )
            actual = path.stat().st_size
            if actual != expected:
                raise SystemExit(
                    f"[error] shard {path.name} size is {actual}, expected {expected}"
                )

    assembled_dir = workdir / "assembled"
    assembled_dir.mkdir(parents=True, exist_ok=True)
    zip_path = assembled_dir / zip_name

    print(f"rebuilding_split_zip={zip_path}")
    with zip_path.open("wb") as out:
        for path in shard_paths:
            print(f"  shard={path.name} size={path.stat().st_size}")
            with path.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)

    actual_zip_size = zip_path.stat().st_size
    if actual_zip_size != expected_zip_size:
        raise SystemExit(
            f"[error] rebuilt zip size is {actual_zip_size}, expected {expected_zip_size}"
        )
    if not zipfile.is_zipfile(zip_path):
        raise SystemExit(f"[error] rebuilt file is not a valid zip: {zip_path}")
    return zip_path


split_zip = rebuild_split_zip()
archive_roots = [split_zip] if split_zip is not None else list(download_dir.rglob("*"))

archive_count = 0
for path in archive_roots:
    if path is None:
        continue
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
