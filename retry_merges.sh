#!/bin/bash
# Retry merge for groups that failed with OOM
# Runs merge_predictions.py sequentially (MERGE_JOBS=1) to avoid OOM
#
# Usage:
#   bash retry_merges.sh              # dry-run (default)
#   bash retry_merges.sh --execute    # actually run merges

MODE="${1:---dry-run}"
SCRIPT_DIR="/home/nibio/mutable-outside-world"
OUTPUT_DIR="/home/mv_out"
FINAL_DIR="$OUTPUT_DIR/final_results"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

mkdir -p "$FINAL_DIR"

# OOM groups: predictions exist but merge failed with TerminatedWorkerError
OOM_GROUPS=""
for g in $(seq 0 159); do
    WL="$OUTPUT_DIR/group_$g/worker.log"
    if [[ -f "$WL" ]] && grep -q "TerminatedWorkerError" "$WL" 2>/dev/null; then
        OOM_GROUPS="$OOM_GROUPS $g"
    fi
done

echo "=== Retry Merge for OOM Groups ==="
echo "Groups to retry: $OOM_GROUPS"
echo "Using MERGE_JOBS=1 (sequential) to avoid OOM"
echo ""

SUCCESS=0
FAILED=0

for g in $OOM_GROUPS; do
    GROUP_DIR="$OUTPUT_DIR/group_$g"
    EVAL_YAML="$GROUP_DIR/eval.yaml"

    if [[ ! -f "$EVAL_YAML" ]]; then
        echo "SKIP group_$g: no eval.yaml"
        continue
    fi

    # Check predictions exist
    PRED_COUNT=$(ls "$GROUP_DIR"/predictions_*.npz 2>/dev/null | wc -l)
    INPUT_DIR="$OUTPUT_DIR/group_${g}_input"
    EXPECTED=$(ls "$INPUT_DIR"/*.ply 2>/dev/null | wc -l)

    if [[ "$PRED_COUNT" -lt "$EXPECTED" ]]; then
        echo "SKIP group_$g: incomplete predictions ($PRED_COUNT/$EXPECTED) - needs re-inference, not just re-merge"
        continue
    fi

    echo "MERGE group_$g: $PRED_COUNT predictions -> $FINAL_DIR"

    if [[ "$MODE" == "--execute" ]]; then
        MERGE_JOBS=1 python3 "$SCRIPT_DIR/nibio_inference/merge_predictions.py" \
            -e "$EVAL_YAML" \
            -p "$GROUP_DIR" \
            -o "$FINAL_DIR" \
            -v 2>&1 | tee -a "$GROUP_DIR/worker.log"

        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            echo "OK group_$g: merge succeeded"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "FAIL group_$g: merge failed again"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    fi
done

echo ""
echo "=== Summary ==="
if [[ "$MODE" != "--execute" ]]; then
    echo "MODE: DRY RUN (use --execute to actually run merges)"
else
    echo "Succeeded: $SUCCESS"
    echo "Failed:    $FAILED"
fi
