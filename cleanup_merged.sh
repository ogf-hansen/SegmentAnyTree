#!/bin/bash
# Cleanup script for merged groups
# Removes utm2local source files and group prediction dirs for groups that:
#   1. Have "Merge complete" in the main inference log
#   2. Have NO errors/exceptions in their worker.log
#   3. Have not been cleaned already
#
# Usage: bash /home/mv_out/cleanup_merged.sh [--dry-run]
#   --dry-run  Show what would be removed without deleting anything

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No files will be deleted."
    echo ""
fi

LOG_FILE="/home/mv_out/inference_20260324_175758.log"
OUTPUT_DIR="/home/mv_out"

if [[ ! -f "$LOG_FILE" ]]; then
    echo "ERROR: Inference log not found: $LOG_FILE"
    exit 1
fi

# Get merged groups from main log
MERGED=$(grep "Merge complete" "$LOG_FILE" | grep -oP "Group \K\d+" | sort -n | uniq)

CLEANED=0
SKIPPED_NODATA=0
SKIPPED_ERRORS=0
FREED_BYTES=0

for g in $MERGED; do
    GROUP_DIR="$OUTPUT_DIR/group_${g}"
    INPUT_DIR="$OUTPUT_DIR/group_${g}_input"

    # Skip if already cleaned
    if [[ ! -d "$GROUP_DIR" ]] && [[ ! -d "$INPUT_DIR" ]]; then
        SKIPPED_NODATA=$((SKIPPED_NODATA + 1))
        continue
    fi

    # Check worker log for errors - if ANY errors, skip this group
    WORKER_LOG="$GROUP_DIR/worker.log"
    if [[ -f "$WORKER_LOG" ]]; then
        ERROR_COUNT=$(grep -ci "error\|Traceback\|Exception\|SIGSEGV\|Killed\|TerminatedWorker" "$WORKER_LOG" 2>/dev/null || true)
        if [[ "$ERROR_COUNT" -gt 0 ]]; then
            echo "SKIP group_$g: $ERROR_COUNT error(s) found in worker.log (keeping predictions)"
            SKIPPED_ERRORS=$((SKIPPED_ERRORS + 1))
            continue
        fi
    fi

    # Calculate size before removal
    GROUP_SIZE=0
    if [[ -d "$GROUP_DIR" ]]; then
        GROUP_SIZE=$(du -sb "$GROUP_DIR" 2>/dev/null | awk '{print $1}')
    fi

    UTM_SIZE=0
    UTM_FILES=""
    if [[ -d "$INPUT_DIR" ]]; then
        for f in "$INPUT_DIR"/*; do
            [[ -L "$f" ]] || continue
            TARGET=$(readlink -f "$f")
            if [[ -f "$TARGET" ]]; then
                S=$(stat -c%s "$TARGET" 2>/dev/null || echo 0)
                UTM_SIZE=$((UTM_SIZE + S))
                UTM_FILES="$UTM_FILES $TARGET"
            fi
        done
    fi

    TOTAL=$((GROUP_SIZE + UTM_SIZE))
    FREED_BYTES=$((FREED_BYTES + TOTAL))

    if $DRY_RUN; then
        echo "WOULD CLEAN group_$g: ~$((TOTAL / 1024 / 1024)) MB"
    else
        # Remove utm2local source files (the actual data the symlinks point to)
        for TARGET in $UTM_FILES; do
            rm -f "$TARGET"
        done
        # Remove input symlink dir
        rm -rf "$INPUT_DIR"
        # Remove predictions dir
        rm -rf "$GROUP_DIR"
        echo "CLEANED group_$g: ~$((TOTAL / 1024 / 1024)) MB freed"
    fi

    CLEANED=$((CLEANED + 1))
done

echo ""
echo "=== Summary ==="
echo "Cleaned:             $CLEANED groups"
echo "Skipped (already):   $SKIPPED_NODATA groups"
echo "Skipped (errors):    $SKIPPED_ERRORS groups"
echo "Space freed:         $((FREED_BYTES / 1024 / 1024 / 1024)) GB ($((FREED_BYTES / 1024 / 1024)) MB)"
echo ""
df -h /home
