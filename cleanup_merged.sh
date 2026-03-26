#!/bin/bash
# Cleanup script for merged groups
# Removes utm2local source files and group prediction dirs for groups that:
#   1. Have "Merge complete" in the main inference log
#   2. Have NO errors/exceptions in their worker.log
#   3. Have complete predictions (all expected .npz files present)
#   4. Have not been cleaned already
#
# Usage:
#   bash cleanup_merged.sh              # dry-run (default, safe)
#   bash cleanup_merged.sh --execute    # actually delete files
#   bash cleanup_merged.sh --status     # show full status of all groups

set -euo pipefail

MODE="${1:---dry-run}"
LOG_FILE="/home/mv_out/inference_20260326_013617.log"
OUTPUT_DIR="/home/mv_out"

if [[ ! -f "$LOG_FILE" ]]; then
    echo "ERROR: Inference log not found: $LOG_FILE"
    exit 1
fi

# Get merged groups from main log
MERGED=$(grep "Merge complete" "$LOG_FILE" | grep -oP "Group \K\d+" | sort -n | uniq)

# Get all launched groups
LAUNCHED=$(grep "Launched group" "$LOG_FILE" | grep -oP "Launched group \K\d+" | sort -n | uniq)

if [[ "$MODE" == "--status" ]]; then
    echo "=== Full Group Status ==="
    echo ""
    for g in $(echo "$LAUNCHED" | sort -n); do
        GROUP_DIR="$OUTPUT_DIR/group_${g}"
        INPUT_DIR="$OUTPUT_DIR/group_${g}_input"
        WL="$GROUP_DIR/worker.log"

        IS_MERGED=$(echo "$MERGED" | grep -qw "$g" && echo "YES" || echo "NO")

        PREDICTIONS=0
        EXPECTED=0
        if [[ -d "$INPUT_DIR" ]]; then
            EXPECTED=$(ls "$INPUT_DIR"/*.ply 2>/dev/null | wc -l)
        fi
        if [[ -d "$GROUP_DIR" ]]; then
            PREDICTIONS=$(ls "$GROUP_DIR"/predictions_*.npz 2>/dev/null | wc -l)
        fi

        HAS_ERRORS="NO"
        ERROR_TYPE=""
        if [[ -f "$WL" ]]; then
            if grep -q "TerminatedWorkerError" "$WL" 2>/dev/null; then
                HAS_ERRORS="YES"
                ERROR_TYPE="OOM/SIGSEGV"
            elif grep -qi "Traceback\|Exception" "$WL" 2>/dev/null; then
                HAS_ERRORS="YES"
                ERROR_TYPE="other"
            fi
        fi

        CLEANED="NO"
        if [[ ! -d "$GROUP_DIR" ]] && [[ ! -d "$INPUT_DIR" ]]; then
            CLEANED="YES"
        fi

        if [[ "$CLEANED" == "YES" ]]; then
            STATUS="CLEANED"
        elif [[ "$IS_MERGED" == "YES" ]] && [[ "$HAS_ERRORS" == "NO" ]]; then
            STATUS="SAFE_TO_CLEAN"
        elif [[ "$HAS_ERRORS" == "YES" ]]; then
            STATUS="ERROR($ERROR_TYPE)"
        elif [[ "$PREDICTIONS" -lt "$EXPECTED" ]] && [[ "$EXPECTED" -gt 0 ]]; then
            STATUS="INCOMPLETE_INF($PREDICTIONS/$EXPECTED)"
        else
            STATUS="NO_MERGE"
        fi

        echo "  group_$g: $STATUS"
    done

    echo ""
    df -h /home
    exit 0
fi

# Main cleanup logic
CLEANED=0
SKIPPED_NODATA=0
SKIPPED_ERRORS=0
SKIPPED_INCOMPLETE=0
FREED_BYTES=0

for g in $MERGED; do
    GROUP_DIR="$OUTPUT_DIR/group_${g}"
    INPUT_DIR="$OUTPUT_DIR/group_${g}_input"

    # Skip if already cleaned
    if [[ ! -d "$GROUP_DIR" ]] && [[ ! -d "$INPUT_DIR" ]]; then
        SKIPPED_NODATA=$((SKIPPED_NODATA + 1))
        continue
    fi

    # Check worker log for errors - if ANY errors, skip (keep predictions for re-merge)
    WORKER_LOG="$GROUP_DIR/worker.log"
    if [[ -f "$WORKER_LOG" ]]; then
        ERROR_COUNT=$(grep -ci "TerminatedWorkerError\|Traceback\|SIGSEGV\|Killed" "$WORKER_LOG" 2>/dev/null || true)
        if [[ "$ERROR_COUNT" -gt 0 ]]; then
            echo "SKIP group_$g: errors in worker.log (keeping predictions for re-merge)"
            SKIPPED_ERRORS=$((SKIPPED_ERRORS + 1))
            continue
        fi
    fi

    # Check predictions are complete
    if [[ -d "$INPUT_DIR" ]] && [[ -d "$GROUP_DIR" ]]; then
        EXPECTED=$(ls "$INPUT_DIR"/*.ply 2>/dev/null | wc -l)
        PREDICTIONS=$(ls "$GROUP_DIR"/predictions_*.npz 2>/dev/null | wc -l)
        if [[ "$PREDICTIONS" -lt "$EXPECTED" ]] && [[ "$EXPECTED" -gt 0 ]]; then
            echo "SKIP group_$g: incomplete predictions ($PREDICTIONS/$EXPECTED) - needs re-inference"
            SKIPPED_INCOMPLETE=$((SKIPPED_INCOMPLETE + 1))
            continue
        fi
    fi

    # Calculate size
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

    if [[ "$MODE" != "--execute" ]]; then
        echo "WOULD CLEAN group_$g: ~$((TOTAL / 1024 / 1024)) MB"
    else
        for TARGET in $UTM_FILES; do
            rm -f "$TARGET"
        done
        rm -rf "$INPUT_DIR"
        rm -rf "$GROUP_DIR"
        echo "CLEANED group_$g: ~$((TOTAL / 1024 / 1024)) MB freed"
    fi

    CLEANED=$((CLEANED + 1))
done

echo ""
echo "=== Summary ==="
if [[ "$MODE" != "--execute" ]]; then
    echo "MODE: DRY RUN (use --execute to actually delete)"
fi
echo "Cleanable:               $CLEANED groups"
echo "Already cleaned:         $SKIPPED_NODATA groups"
echo "Skipped (errors):        $SKIPPED_ERRORS groups (need re-merge)"
echo "Skipped (incomplete):    $SKIPPED_INCOMPLETE groups (need re-inference)"
echo "Space to free:           $((FREED_BYTES / 1024 / 1024 / 1024)) GB ($((FREED_BYTES / 1024 / 1024)) MB)"
echo ""
df -h /home
