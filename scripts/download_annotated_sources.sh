#!/usr/bin/env bash
# Download the source videos named in a CVAT export manifest into .local/source_videos/.
# Idempotent: files already present at the correct size are skipped.
# Downloads run PARALLEL_FILES at a time — a single S3 stream is bandwidth-limited here,
# but too many concurrent transfers exhaust the CLI's own retries on this link, so each
# file is also retried with backoff.
set -Eeuo pipefail

MANIFEST="${1:-.local/cvat_exports/2026-09-09/export_manifest.csv}"
DEST="${2:-.local/source_videos}"
PARALLEL_FILES="${PARALLEL_FILES:-3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
BUCKET="chillnbite-cameras"
PREFIX="anon"

set -a; . .local/env/storage.env; set +a
mkdir -p "$DEST"

fetch_one() {
    local name="$1" dest="$2" bucket="$3" prefix="$4"
    local remote local_size
    remote=$(aws s3api head-object --bucket "$bucket" --key "$prefix/$name" --query ContentLength --output text)
    if [ -f "$dest/$name" ]; then
        local_size=$(stat -f%z "$dest/$name")
        if [ "$remote" = "$local_size" ]; then
            echo "skip  $name"
            return 0
        fi
    fi
    local attempt=1
    while [ "$attempt" -le "${MAX_ATTEMPTS:-4}" ]; do
        if aws s3 cp "s3://$bucket/$prefix/$name" "$dest/$name" --only-show-errors; then
            local_size=$(stat -f%z "$dest/$name" 2>/dev/null || echo 0)
            if [ "$remote" = "$local_size" ]; then
                echo "ok    $name"
                return 0
            fi
            echo "retry $name (size $local_size != $remote)" >&2
        else
            echo "retry $name (attempt $attempt failed)" >&2
        fi
        rm -f "$dest/$name" "$dest/$name".*
        sleep $((attempt * 10))
        attempt=$((attempt + 1))
    done
    echo "FAIL  $name after ${MAX_ATTEMPTS:-4} attempts" >&2
    return 1
}
export -f fetch_one
export MAX_ATTEMPTS

tail -n +2 "$MANIFEST" | cut -d, -f3 | sort -u \
    | xargs -P "$PARALLEL_FILES" -I{} bash -c 'fetch_one "$@"' _ {} "$DEST" "$BUCKET" "$PREFIX"

echo "done: $(ls -1 "$DEST"/*.mp4 2>/dev/null | wc -l) files, $(du -sh "$DEST" | cut -f1)"
