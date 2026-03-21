#!/bin/bash
set -e

# Parallel inference script for SegmentAnyTree
# Splits tiles into groups and runs multiple eval.py processes concurrently across GPUs.
# Usage: run_parallel_inference.sh <input_dir> <output_dir> [clean_output_dir]
# Environment variables:
#   N_GPU_WORKERS  - number of concurrent eval.py processes (default: 2)
#   N_GPUS         - number of GPUs available; workers are distributed round-robin (default: 1)
#   STAGGER_DELAY  - seconds between process launches (default: 30)

SOURCE_DIR="$1"
DEST_DIR="$2"
CLEAN_OUTPUT_DIR="$3"

N_GPU_WORKERS="${N_GPU_WORKERS:-2}"
N_GPUS="${N_GPUS:-1}"
STAGGER_DELAY="${STAGGER_DELAY:-30}"

# Set default values if not provided
: "${SOURCE_DIR:=/home/nibio/mutable-outside-world/data_for_test}"
: "${DEST_DIR:=/home/nibio/mutable-outside-world/data_for_test_results}"
: "${CLEAN_OUTPUT_DIR:=true}"

if [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ] || [ -z "$CLEAN_OUTPUT_DIR" ]; then
    echo "Usage: run_parallel_inference.sh <path_to_input_dir> <path_to_output_dir> <clean_output_dir>"
    echo "Environment variables: N_GPU_WORKERS (default: 2), N_GPUS (default: 1), STAGGER_DELAY (default: 30)"
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

# ============================================================
# Logging and timing setup
# ============================================================
mkdir -p "$DEST_DIR"
LOG_FILE="$DEST_DIR/inference_$(date +%Y%m%d_%H%M%S).log"
TOTAL_START=$SECONDS

# Redirect all stdout and stderr to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

log_phase_time() {
    local phase_name="$1"
    local start="$2"
    local elapsed=$(( SECONDS - start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo "[TIMING] $phase_name: ${mins}m ${secs}s"
}

echo "=== Parallel Inference ==="
echo "Started at: $(date)"
echo "Input directory: $SOURCE_DIR"
echo "Output directory: $DEST_DIR"
echo "GPU workers: $N_GPU_WORKERS"
echo "GPUs available: $N_GPUS"
echo "Stagger delay: ${STAGGER_DELAY}s"
echo "Log file: $LOG_FILE"

# ============================================================
# PHASE 1: Shared Preprocessing (runs once)
# ============================================================
echo ""
echo "=== Phase 1: Preprocessing ==="
PHASE_START=$SECONDS

# Copy input files and fix naming
mkdir -p "$DEST_DIR/input_data"
for f in "$SOURCE_DIR"/*; do
    ln -sf "$f" "$DEST_DIR/input_data/"
done

python3 "$SCRIPT_DIR/nibio_inference/fix_naming_of_input_files.py" "$DEST_DIR/input_data"

# UTM normalization (uses all CPU cores via n_jobs=-1)
python3 "$SCRIPT_DIR/nibio_inference/pipeline_utm2local_parallel.py" -i "$DEST_DIR/input_data" -o "$DEST_DIR/utm2local"

# Create a base eval.yaml for cache clearing
cp "$SCRIPT_DIR/conf/eval.yaml" "$DEST_DIR"
python3 "$SCRIPT_DIR/nibio_inference/modify_eval.py" "$DEST_DIR/eval.yaml" "$DEST_DIR/utm2local" "$DEST_DIR"

# Clear cache once before any inference
python3 "$SCRIPT_DIR/nibio_inference/clear_cache.py" --eval_yaml "$DEST_DIR/eval.yaml"

log_phase_time "Phase 1 (Preprocessing)" $PHASE_START

# ============================================================
# PHASE 2: Split files into groups
# ============================================================
echo ""
echo "=== Phase 2: Splitting into $N_GPU_WORKERS groups ==="
PHASE_START=$SECONDS

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
        # Also symlink the corresponding _min_values.json needed by merge_predictions
        JSON_FILE="${PLY_FILES[$i]%.ply}_min_values.json"
        if [ -f "$JSON_FILE" ]; then
            ln -sf "$JSON_FILE" "$GROUP_INPUT_DIR/"
        fi
        COUNT=$((COUNT + 1))
    done

    # Skip empty groups (can happen if files < workers)
    if [ "$COUNT" -eq 0 ]; then
        echo "Group $k: 0 files (skipped)"
        continue
    fi

    echo "Group $k: $COUNT files -> GPU $((k % N_GPUS))"

    # Create group-specific eval.yaml with its own fold list and output dir
    cp "$SCRIPT_DIR/conf/eval.yaml" "$GROUP_OUTPUT_DIR/eval.yaml"
    python3 "$SCRIPT_DIR/nibio_inference/modify_eval.py" \
        "$GROUP_OUTPUT_DIR/eval.yaml" \
        "$GROUP_INPUT_DIR" \
        "$GROUP_OUTPUT_DIR"
done

log_phase_time "Phase 2 (Splitting)" $PHASE_START

# ============================================================
# PHASE 3: Parallel GPU Inference
# ============================================================
echo ""
echo "=== Phase 3: Running $N_GPU_WORKERS parallel inference processes across $N_GPUS GPU(s) ==="
PHASE_START=$SECONDS

PIDS=()
WORKER_STARTS=()
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

    GPU_ID=$((k % N_GPUS))
    WORKER_STARTS+=($SECONDS)

    (
        echo "[Group $k] Starting inference on $FILE_COUNT files on GPU $GPU_ID at $(date +%H:%M:%S)..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 eval.py --config-name "$GROUP_OUTPUT_DIR/eval.yaml"
        echo "[Group $k] Inference complete at $(date +%H:%M:%S)."
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
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if wait "$pid"; then
        elapsed=$(( SECONDS - WORKER_STARTS[i] ))
        mins=$(( elapsed / 60 ))
        secs=$(( elapsed % 60 ))
        echo "[TIMING] Worker $i (PID $pid): ${mins}m ${secs}s"
    else
        echo "ERROR: Process $pid failed"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED group(s) failed. Check logs above."
    exit 1
fi

echo "All groups completed successfully."
log_phase_time "Phase 3 (Inference total)" $PHASE_START

# ============================================================
# PHASE 4: Collect results and merge
# ============================================================
echo ""
echo "=== Phase 4: Collecting results and merging ==="
PHASE_START=$SECONDS

FINAL_DEST_DIR="$DEST_DIR/final_results"

# Merge predictions from each group using their group-specific eval.yaml
for ((k=0; k<N_GPU_WORKERS; k++)); do
    GROUP_OUTPUT_DIR="$DEST_DIR/group_${k}"
    if [ -f "$GROUP_OUTPUT_DIR/eval.yaml" ]; then
        echo "Merging predictions from group $k..."
        python3 "$SCRIPT_DIR/nibio_inference/merge_predictions.py" \
            -e "$GROUP_OUTPUT_DIR/eval.yaml" \
            -p "$GROUP_OUTPUT_DIR" \
            -o "$FINAL_DEST_DIR" \
            -v
    fi
done

log_phase_time "Phase 4 (Merging)" $PHASE_START

num_files=$(find "$FINAL_DEST_DIR" -maxdepth 1 -type f | wc -l)

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "=== Complete ==="
echo "Finished at: $(date)"
echo "[TIMING] Total elapsed: ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "Number of files in final results: $num_files"
echo "Results directory: $FINAL_DEST_DIR"
echo "Log file: $LOG_FILE"
