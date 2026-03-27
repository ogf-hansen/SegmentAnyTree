#!/bin/bash
# Remerge groups after inference.
#
# Usage:
#   bash remerge_groups.sh <output_dir> <group_number>    # single group
#   bash remerge_groups.sh <output_dir> --missing          # only groups with missing output
#   bash remerge_groups.sh <output_dir> --all              # remerge every group
#   bash remerge_groups.sh <output_dir> --status           # show status only

set -uo pipefail

SCRIPT_DIR="/home/nibio/mutable-outside-world"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

OUTPUT_DIR="${1:?Usage: remerge_groups.sh <output_dir> <group | --missing | --all | --status>}"
MODE="${2:?Specify a group number, --missing, --all, or --status}"

FINAL_DIR="$OUTPUT_DIR/final_results"
mkdir -p "$FINAL_DIR"

# ---------------------------------------------------------------
# Get expected output basenames for a group
# ---------------------------------------------------------------
get_expected_bases() {
    local g="$1"
    python3 -c "
import yaml, os
with open('$OUTPUT_DIR/group_${g}/eval.yaml') as f:
    fold = yaml.safe_load(f).get('data',{}).get('fold',[])
for p in fold:
    b = os.path.splitext(os.path.basename(p))[0]
    if b.endswith('_out'): b = b[:-4]
    print(b)
" 2>/dev/null
}

# ---------------------------------------------------------------
# List all valid group numbers
# ---------------------------------------------------------------
all_groups() {
    for d in "$OUTPUT_DIR"/group_*_input; do
        [[ -d "$d" ]] || continue
        basename "$d" | sed 's/group_\([0-9]*\)_input/\1/'
    done | sort -n
}

# ---------------------------------------------------------------
# Check if a group has missing output
# ---------------------------------------------------------------
group_is_missing() {
    local g="$1"
    while IFS= read -r base; do
        if [[ ! -f "$FINAL_DIR/${base}.las" ]] && [[ ! -f "$FINAL_DIR/${base}.laz" ]]; then
            return 0
        fi
    done < <(get_expected_bases "$g")
    return 1
}

# ---------------------------------------------------------------
# Merge a single group
# ---------------------------------------------------------------
merge_group() {
    local g="$1"
    local eval_yaml="$OUTPUT_DIR/group_${g}/eval.yaml"
    local npz_count

    if [[ ! -f "$eval_yaml" ]]; then
        echo "[Group $g] SKIP: no eval.yaml"
        return 1
    fi

    npz_count=$(find "$OUTPUT_DIR/group_${g}" -maxdepth 1 -name "predictions_*.npz" 2>/dev/null | wc -l)
    if [[ "$npz_count" -eq 0 ]]; then
        echo "[Group $g] SKIP: no predictions (needs re-inference)"
        return 1
    fi

    echo "[Group $g] Remerging..."
    python3 "$SCRIPT_DIR/nibio_inference/merge_predictions.py" \
        -e "$eval_yaml" \
        -p "$OUTPUT_DIR/group_${g}" \
        -o "$FINAL_DIR" \
        -v
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
case "$MODE" in
    --status)
        echo "=== Group Merge Status ==="
        for g in $(all_groups); do
            npz=$(find "$OUTPUT_DIR/group_${g}" -maxdepth 1 -name "predictions_*.npz" 2>/dev/null | wc -l)
            if group_is_missing "$g" && [[ "$npz" -gt 0 ]]; then
                echo "  Group $g: NEEDS_REMERGE"
            elif group_is_missing "$g"; then
                echo "  Group $g: NEEDS_REINFERENCE"
            else
                echo "  Group $g: OK"
            fi
        done
        echo ""
        echo "Files in final_results: $(ls "$FINAL_DIR" 2>/dev/null | wc -l)"
        ;;

    --missing)
        echo "=== Remerging groups with missing output ==="
        count=0
        for g in $(all_groups); do
            if group_is_missing "$g"; then
                merge_group "$g" && count=$((count + 1))
            fi
        done
        echo ""
        echo "Remerged: $count groups"
        echo "Files in final_results: $(ls "$FINAL_DIR" 2>/dev/null | wc -l)"
        ;;

    --all)
        echo "=== Remerging all groups ==="
        count=0
        for g in $(all_groups); do
            merge_group "$g" && count=$((count + 1))
        done
        echo ""
        echo "Remerged: $count groups"
        echo "Files in final_results: $(ls "$FINAL_DIR" 2>/dev/null | wc -l)"
        ;;

    *)
        merge_group "$MODE"
        ;;
esac
