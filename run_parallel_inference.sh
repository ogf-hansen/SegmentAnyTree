#!/bin/bash
set -e

# Parallel inference script for SegmentAnyTree
# Splits tiles into groups and runs multiple eval.py processes concurrently on a single GPU.
# Usage: run_parallel_inference.sh <input_dir> <output_dir> [clean_output_dir]
# Environment variables:
#   N_GPU_WORKERS  - number of concurrent eval.py processes (default: 2)
#   STAGGER_DELAY  - seconds between process launches (default: 30)

SOURCE_DIR="$1"
DEST_DIR="$2"
CLEAN_OUTPUT_DIR="$3"

N_GPU_WORKERS="${N_GPU_WORKERS:-2}"
STAGGER_DELAY="${STAGGER_DELAY:-30}"

# Set default values if not provided
: "${SOURCE_DIR:=/home/nibio/mutable-outside-world/data_for_test}"
: "${DEST_DIR:=/home/nibio/mutable-outside-world/data_for_test_results}"
: "${CLEAN_OUTPUT_DIR:=true}"

if [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ] || [ -z "$CLEAN_OUTPUT_DIR" ]; then
    echo "Usage: run_parallel_inference.sh <path_to_input_dir> <path_to_output_dir> <clean_output_dir>"
    echo "Environment variables: N_GPU_WORKERS (default: 2), STAGGER_DELAY (default: 30)"
    exit 1
fi

# Clear the output directory if requested
if [ "$CLEAN_OUTPUT_DIR" = "true" ]; then
    rm -rf "$DEST_DIR"/*
fi

# Convert relative paths to absolute
if [[ "$SOURCE_DIR" != /* ]]; then
    SOURCE_DIR=$(pwd)/"$SOURCE_DIR"
fi

if [[ "$DEST_DIR" != /* ]]; then
    DEST_DIR=$(pwd)/"$DEST_DIR"
fi

# Set the script directory and add it to the PYTHONPATH
SCRIPT_DIR="/home/nibio/mutable-outside-world"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "=== Parallel Inference ==="
echo "Input directory: $SOURCE_DIR"
echo "Output directory: $DEST_DIR"
echo "GPU workers: $N_GPU_WORKERS"
echo "Stagger delay: ${STAGGER_DELAY}s"

# ============================================================
# PHASE 1: Shared Preprocessing (runs once)
# ============================================================
echo ""
echo "=== Phase 1: Preprocessing ==="

# Copy input files and fix naming
mkdir -p "$DEST_DIR/input_data"
cp -r "$SOURCE_DIR/"* "$DEST_DIR/input_data/"

python3 "$SCRIPT_DIR/nibio_inference/fix_naming_of_input_files.py" "$DEST_DIR/input_data"

# UTM normalization (uses all CPU cores via n_jobs=-1)
python3 "$SCRIPT_DIR/nibio_inference/pipeline_utm2local_parallel.py" -i "$DEST_DIR/input_data" -o "$DEST_DIR/utm2local"

# Create a base eval.yaml for cache clearing
cp "$SCRIPT_DIR/conf/eval.yaml" "$DEST_DIR"
python3 "$SCRIPT_DIR/nibio_inference/modify_eval.py" "$DEST_DIR/eval.yaml" "$DEST_DIR/utm2local" "$DEST_DIR"

# Clear cache once before any inference
python3 "$SCRIPT_DIR/nibio_inference/clear_cache.py" --eval_yaml "$DEST_DIR/eval.yaml"

# ============================================================
# PHASE 2: Split files into groups
# ============================================================
echo ""
echo "=== Phase 2: Splitting into $N_GPU_WORKERS groups ==="

# Get all .ply files from utm2local
mapfile -t PLY_FILES < <(find "$DEST_DIR/utm2local" -maxdepth 1 -name "*.ply" -type f | sort)
TOTAL_FILES=${#PLY_FILES[@]}

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "ERROR: No .ply files found in $DEST_DIR/utm2local"
    exit 1
fi

echo "Total files: $TOTAL_FILES"

# Calculate files per group (ceiling division)
FILES_PER_GROUP=$(( (TOTAL_FILES + N_GPU_WORKERS - 1) / N_GPU_WORKERS ))

# Create group directories with symlinks to their subset of files
for ((k=0; k<N_GPU_WORKERS; k++)); do
    GROUP_INPUT_DIR="$DEST_DIR/group_${k}_input"
    GROUP_OUTPUT_DIR="$DEST_DIR/group_${k}"
    mkdir -p "$GROUP_INPUT_DIR"
    mkdir -p "$GROUP_OUTPUT_DIR"

    START=$((k * FILES_PER_GROUP))
    END=$((START + FILES_PER_GROUP))
    if [ "$END" -gt "$TOTAL_FILES" ]; then
        END=$TOTAL_FILES
    fi

    COUNT=0
    for ((i=START; i<END; i++)); do
        ln -sf "${PLY_FILES[$i]}" "$GROUP_INPUT_DIR/"
        COUNT=$((COUNT + 1))
    done

    # Skip empty groups (can happen if files < workers)
    if [ "$COUNT" -eq 0 ]; then
        echo "Group $k: 0 files (skipped)"
        continue
    fi

    echo "Group $k: $COUNT files"

    # Create group-specific eval.yaml with its own fold list and output dir
    cp "$SCRIPT_DIR/conf/eval.yaml" "$GROUP_OUTPUT_DIR/eval.yaml"
    python3 "$SCRIPT_DIR/nibio_inference/modify_eval.py" \
        "$GROUP_OUTPUT_DIR/eval.yaml" \
        "$GROUP_INPUT_DIR" \
        "$GROUP_OUTPUT_DIR"
done

# ============================================================
# PHASE 3: Parallel GPU Inference
# ============================================================
echo ""
echo "=== Phase 3: Running $N_GPU_WORKERS parallel inference processes ==="

PIDS=()
for ((k=0; k<N_GPU_WORKERS; k++)); do
    GROUP_OUTPUT_DIR="$DEST_DIR/group_${k}"
    GROUP_INPUT_DIR="$DEST_DIR/group_${k}_input"

    # Skip groups with no files
    if [ ! -f "$GROUP_OUTPUT_DIR/eval.yaml" ]; then
        continue
    fi

    # Check if group has any files in its fold
    FILE_COUNT=$(find "$GROUP_INPUT_DIR" -maxdepth 1 -name "*.ply" -type l 2>/dev/null | wc -l)
    if [ "$FILE_COUNT" -eq 0 ]; then
        continue
    fi

    (
        echo "[Group $k] Starting inference on $FILE_COUNT files..."
        python3 eval.py --config-name "$GROUP_OUTPUT_DIR/eval.yaml"

        echo "[Group $k] Inference complete. Renaming result files..."
        python3 "$SCRIPT_DIR/nibio_inference/rename_result_files_instance.py" \
            "$GROUP_OUTPUT_DIR/eval.yaml" "$GROUP_OUTPUT_DIR"
        python3 "$SCRIPT_DIR/nibio_inference/rename_result_files_segmentation.py" \
            "$GROUP_OUTPUT_DIR/eval.yaml" "$GROUP_OUTPUT_DIR"

        echo "[Group $k] Done."
    ) &
    PIDS+=($!)

    # Stagger launches to avoid simultaneous GPU memory peaks
    if [ "$k" -lt $((N_GPU_WORKERS - 1)) ]; then
        echo "Waiting ${STAGGER_DELAY}s before launching next group..."
        sleep "$STAGGER_DELAY"
    fi
done

# Wait for all groups to finish
echo "Waiting for all groups to finish..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Process $pid failed"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED group(s) failed. Check logs above."
    exit 1
fi

echo "All groups completed successfully."

# ============================================================
# PHASE 4: Collect results and merge
# ============================================================
echo ""
echo "=== Phase 4: Collecting results and merging ==="

# Collect all renamed segmentation files from group dirs into DEST_DIR
for ((k=0; k<N_GPU_WORKERS; k++)); do
    GROUP_OUTPUT_DIR="$DEST_DIR/group_${k}"
    for f in "$GROUP_OUTPUT_DIR"/instance_segmentation_*.ply; do
        [ -f "$f" ] && mv "$f" "$DEST_DIR/"
    done
    for f in "$GROUP_OUTPUT_DIR"/semantic_segmentation_*.ply; do
        [ -f "$f" ] && mv "$f" "$DEST_DIR/"
    done
done

FINAL_DEST_DIR="$DEST_DIR/final_results"

# Run parallel merge
python3 "$SCRIPT_DIR/nibio_inference/merge_pt_ss_is_in_folders_parallel.py" \
    -i "$DEST_DIR/utm2local" -s "$DEST_DIR" -o "$FINAL_DEST_DIR" -v

# Remove number prefixes from final file names (e.g., "0_filename.las" -> "filename.las")
for file in "$FINAL_DEST_DIR"/*; do
    filename=$(basename "$file")
    new_name=$(echo "$filename" | sed 's/^[0-9]*_//')
    new_file_path="$FINAL_DEST_DIR/$new_name"
    mv -n "$file" "$new_file_path"
done

num_files=$(find "$FINAL_DEST_DIR" -maxdepth 1 -type f | wc -l)

echo ""
echo "=== Complete ==="
echo "Number of files in final results: $num_files"
echo "Results directory: $FINAL_DEST_DIR"
