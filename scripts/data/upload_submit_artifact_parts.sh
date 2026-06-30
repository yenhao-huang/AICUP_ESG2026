#!/usr/bin/env bash
# Resume-safe Google Drive upload for a large submit artifact zip.
#
# It streams one part at a time from the local zip, verifies the remote size,
# sleeps between parts, and retries failed parts with backoff.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ZIP_PATH="${ZIP_PATH:-/models/aicup_esg2026_submit_artifacts/aicup_esg2026_submit_artifacts.zip}"
PARTS_MANIFEST="${PARTS_MANIFEST:-/models/aicup_esg2026_submit_artifacts/aicup_esg2026_submit_artifacts.zip.parts.txt}"
MANIFEST="${MANIFEST:-/models/aicup_esg2026_submit_artifacts/manifest.txt}"
MANIFEST_DETAILS="${MANIFEST_DETAILS:-/models/aicup_esg2026_submit_artifacts/manifest_details.tsv}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:}"
DRIVE_FOLDER_ID="${DRIVE_FOLDER_ID:-16rutragKIhpiCORINme9-vNvRR8Z3nOt}"
PART_SIZE_MIB="${PART_SIZE_MIB:-3900}"
PART_SLEEP_SECONDS="${PART_SLEEP_SECONDS:-120}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-300}"
PART_RETRIES="${PART_RETRIES:-3}"
START_PART="${START_PART:-0}"
END_PART="${END_PART:-}"
LOG_DIR="${LOG_DIR:-logs/upload_submit_artifacts_parts}"

mkdir -p "$LOG_DIR"

[ -f "$ZIP_PATH" ] || { echo "[error] missing zip: $ZIP_PATH" >&2; exit 1; }
[ -f "$PARTS_MANIFEST" ] || { echo "[error] missing parts manifest: $PARTS_MANIFEST" >&2; exit 1; }
command -v rclone >/dev/null 2>&1 || { echo "[error] missing rclone" >&2; exit 1; }

ZIP_NAME="$(basename "$ZIP_PATH")"
ZIP_SIZE_BYTES="$(stat -Lc '%s' "$ZIP_PATH")"
PART_SIZE_BYTES="$((PART_SIZE_MIB * 1024 * 1024))"
PART_COUNT="$(((ZIP_SIZE_BYTES + PART_SIZE_BYTES - 1) / PART_SIZE_BYTES))"
if [ -z "$END_PART" ]; then
  END_PART="$((PART_COUNT - 1))"
fi

remote_size() {
  local name="$1"
  rclone lsf "$RCLONE_REMOTE" \
    --drive-root-folder-id "$DRIVE_FOLDER_ID" \
    --format ps \
    --files-only 2>/dev/null |
    awk -F ';' -v target="$name" '$1 == target {print $2; found=1} END {if (!found) exit 1}'
}

expected_size_for_part() {
  local index="$1"
  local offset=$((index * PART_SIZE_BYTES))
  local remaining=$((ZIP_SIZE_BYTES - offset))
  if [ "$remaining" -gt "$PART_SIZE_BYTES" ]; then
    echo "$PART_SIZE_BYTES"
  else
    echo "$remaining"
  fi
}

upload_part_once() {
  local index="$1"
  local part_name
  part_name="$(printf '%s.part-%03d' "$ZIP_NAME" "$index")"
  local count_mib="$PART_SIZE_MIB"
  local offset_mib=$((index * PART_SIZE_MIB))

  echo "### uploading $part_name"
  dd if="$ZIP_PATH" bs=1M skip="$offset_mib" count="$count_mib" status=none |
    rclone rcat "${RCLONE_REMOTE}${part_name}" \
      --drive-root-folder-id "$DRIVE_FOLDER_ID" \
      --progress
}

upload_part() {
  local index="$1"
  local part_name expected existing attempt
  part_name="$(printf '%s.part-%03d' "$ZIP_NAME" "$index")"
  expected="$(expected_size_for_part "$index")"

  if existing="$(remote_size "$part_name")" && [ "$existing" = "$expected" ]; then
    echo "### skip $part_name: already uploaded ($existing bytes)"
    return 0
  fi

  attempt=1
  while [ "$attempt" -le "$PART_RETRIES" ]; do
    echo "### part $index/$((PART_COUNT - 1)) attempt $attempt/$PART_RETRIES expected=$expected"
    if upload_part_once "$index"; then
      if existing="$(remote_size "$part_name")" && [ "$existing" = "$expected" ]; then
        echo "### verified $part_name ($existing bytes)"
        return 0
      fi
      echo "[warn] uploaded $part_name but remote size is ${existing:-missing}, expected $expected" >&2
    else
      echo "[warn] upload failed for $part_name" >&2
    fi

    if [ "$attempt" -lt "$PART_RETRIES" ]; then
      echo "### sleeping ${RETRY_SLEEP_SECONDS}s before retry"
      sleep "$RETRY_SLEEP_SECONDS"
    fi
    attempt=$((attempt + 1))
  done

  echo "[error] failed to upload $part_name after $PART_RETRIES attempts" >&2
  return 1
}

upload_small_file() {
  local path="$1"
  [ -f "$path" ] || return 0
  rclone copy "$path" "$RCLONE_REMOTE" --drive-root-folder-id "$DRIVE_FOLDER_ID" --progress
}

echo "### resumable submit artifact parts upload"
echo "zip=$ZIP_PATH"
echo "zip_size_bytes=$ZIP_SIZE_BYTES"
echo "part_size_mib=$PART_SIZE_MIB"
echo "part_count=$PART_COUNT"
echo "range=${START_PART}-${END_PART}"
echo "sleep_between_parts=${PART_SLEEP_SECONDS}s"
echo "remote=$RCLONE_REMOTE folder_id=$DRIVE_FOLDER_ID"

for index in $(seq "$START_PART" "$END_PART"); do
  upload_part "$index" | tee -a "$LOG_DIR/part-$(printf '%03d' "$index").log"
  if [ "$index" -lt "$END_PART" ]; then
    echo "### sleeping ${PART_SLEEP_SECONDS}s before next part"
    sleep "$PART_SLEEP_SECONDS"
  fi
done

echo "### uploading small manifests"
upload_small_file "$PARTS_MANIFEST"
upload_small_file "$MANIFEST"
upload_small_file "$MANIFEST_DETAILS"

echo "### upload DONE"
