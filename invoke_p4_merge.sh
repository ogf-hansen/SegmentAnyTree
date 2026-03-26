#!/bin/bash
DEST_DIR="/home/mv_out"
FINAL_DEST_DIR="$DEST_DIR/final_results"
SCRIPT_DIR="/home/nibio/mutable-outside-world"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

mkdir -p "$FINAL_DEST_DIR"

for k in $(seq 0 179); do
    GROUP_OUTPUT_DIR="$DEST_DIR/group_${k}"
    [ -f "$GROUP_OUTPUT_DIR/eval.yaml" ] || continue

    # Skip groups with no predictions
    NPZ_COUNT=$(find "$GROUP_OUTPUT_DIR" -maxdepth 1 -name "predictions_*.npz" 2>/dev/null | wc -l)
    if [ "$NPZ_COUNT" -eq 0 ]; then
        echo "Group $k: No predictions (no instances), skipping."
        continue
    fi

    # Check for missing outputs (.las or .laz)
    EVAL_YAML="$GROUP_OUTPUT_DIR/eval.yaml"
    EXPECTED=$(python3 -c "
import yaml, os
with open('$EVAL_YAML') as f:
    fold = yaml.safe_load(f).get('data',{}).get('fold',[])
for p in fold:
    b = os.path.splitext(os.path.basename(p))[0]
    if b.endswith('_out'): b = b[:-4]
    print(b)
" 2>/dev/null)

    MISSING=0
    for base in $EXPECTED; do
        if [ ! -f "$FINAL_DEST_DIR/${base}.las" ] && [ ! -f "$FINAL_DEST_DIR/${base}.laz" ]; then
            MISSING=$((MISSING + 1))
        fi
    done

    if [ "$MISSING" -gt 0 ]; then
        echo "Group $k: $MISSING files missing, re-running merge..."
        python3 "$SCRIPT_DIR/nibio_inference/merge_predictions.py" \
            -e "$GROUP_OUTPUT_DIR/eval.yaml" \
            -p "$GROUP_OUTPUT_DIR" \
            -o "$FINAL_DEST_DIR" \
            -v
    fi
done

echo "Done. Total files in final_results: $(ls "$FINAL_DEST_DIR" | wc -l)"
