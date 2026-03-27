#!/bin/bash
set -e

# Parallel inference script for SegmentAnyTree
# Splits tiles into groups and runs multiple eval.py processes concurrently across GPUs.
# Usage: run_parallel_inference.sh <input_dir> <output_dir> [clean_output_dir]
# Environment variables:
#   N_GPU_WORKERS  - total number of groups to split data into (default: 2)
#   N_GPUS         - number of GPUs available (default: 1)
#   WORKERS_PER_GPU - max concurrent workers per GPU (default: 2)

SOURCE_DIR="$1"
DEST_DIR="$2"
CLEAN_OUTPUT_DIR="$3"

N_GPU_WORKERS="${N_GPU_WORKERS:-2}"
N_GPUS="${N_GPUS:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
MAX_CONCURRENT=$((N_GPUS * WORKERS_PER_GPU))

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

# Log helper: writes to both terminal and log file
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log_phase_time() {
    local phase_name="$1"
    local start="$2"
    local elapsed=$(( SECONDS - start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    log "[TIMING] $phase_name: ${mins}m ${secs}s"
}

log "=== Parallel Inference ==="
log "Started at: $(date)"
log "Input directory: $SOURCE_DIR"
log "Output directory: $DEST_DIR"
log "Total groups: $N_GPU_WORKERS"
log "GPUs available: $N_GPUS"
log "Workers per GPU: $WORKERS_PER_GPU (max concurrent: $MAX_CONCURRENT)"
log "Log file: $LOG_FILE"

# ============================================================
# PHASE 1: Shared Preprocessing (runs once)
# ============================================================
log ""
log "=== Phase 1: Preprocessing ==="
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
log ""
log "=== Phase 2: Splitting into $N_GPU_WORKERS groups ==="
PHASE_START=$SECONDS

# Get all .ply files from utm2local
mapfile -t PLY_FILES < <(find "$DEST_DIR/utm2local" -maxdepth 1 -name "*.ply" -type f | sort)
TOTAL_FILES=${#PLY_FILES[@]}

if [ "$TOTAL_FILES" -eq 0 ]; then
    log "ERROR: No .ply files found in $DEST_DIR/utm2local"
    exit 1
fi

log "Total files: $TOTAL_FILES"

# Calculate files per group (ceiling division)
FILES_PER_GROUP=$(( (TOTAL_FILES + N_GPU_WORKERS - 1) / N_GPU_WORKERS ))

# Create group directories with symlinks to their subset of files
for ((k=0; k<N_GPU_WORKERS; k++)); do
    GROUP_INPUT_DIR="$DEST_DIR/group_${k}_input"
    GROUP_OUTPUT_DIR="$DEST_DIR/group_${k}"
    rm -rf "$GROUP_INPUT_DIR" "$GROUP_OUTPUT_DIR"
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
        log "Group $k: 0 files (skipped)"
        continue
    fi

    log "Group $k: $COUNT files -> GPU $((k % N_GPUS))"

    # Create group-specific eval.yaml with its own fold list and output dir
    cp "$SCRIPT_DIR/conf/eval.yaml" "$GROUP_OUTPUT_DIR/eval.yaml"
    python3 "$SCRIPT_DIR/nibio_inference/modify_eval.py" \
        "$GROUP_OUTPUT_DIR/eval.yaml" \
        "$GROUP_INPUT_DIR" \
        "$GROUP_OUTPUT_DIR"
done

log_phase_time "Phase 2 (Splitting)" $PHASE_START

# ============================================================
# PHASE 3: Parallel GPU Inference (worker pool: WORKERS_PER_GPU concurrent per GPU)
# ============================================================
log ""
log "=== Phase 3: Running $N_GPU_WORKERS groups across $N_GPUS GPU(s) ($WORKERS_PER_GPU workers/GPU, $MAX_CONCURRENT concurrent) ==="
PHASE_START=$SECONDS

# Collect valid groups into an array
VALID_GROUPS=()
for ((k=0; k<N_GPU_WORKERS; k++)); do
    GROUP_OUTPUT_DIR="$DEST_DIR/group_${k}"
    GROUP_INPUT_DIR="$DEST_DIR/group_${k}_input"
    if [ ! -f "$GROUP_OUTPUT_DIR/eval.yaml" ]; then
        continue
    fi
    FILE_COUNT=$(find "$GROUP_INPUT_DIR" -maxdepth 1 -name "*.ply" -type l 2>/dev/null | wc -l)
    if [ "$FILE_COUNT" -eq 0 ]; then
        continue
    fi
    VALID_GROUPS+=($k)
done

log "Valid groups: ${#VALID_GROUPS[@]}"

# Cache dir for cleanup (derived from checkpoint's dataroot config)
CACHE_DIR="/home/datascience/tmp_out_folder/utm2local/treeinsfused/processed_0.2_test"

# Helper: launch a group on a given GPU, store PID in slot
launch_group() {
    local k=$1
    local gpu=$2
    local slot=$3
    local group_output="$DEST_DIR/group_${k}"
    local group_input="$DEST_DIR/group_${k}_input"
    local fc
    fc=$(find "$group_input" -maxdepth 1 -name "*.ply" -type l 2>/dev/null | wc -l)

    local group_log="$group_output/worker.log"
    local final_dir="$DEST_DIR/final_results"
    (
        log "[Group $k] Starting inference on $fc files on GPU $gpu at $(date +%H:%M:%S)..."
        CUDA_VISIBLE_DEVICES=$gpu python3 eval.py --config-name "$group_output/eval.yaml" \
            > "$group_log" 2>&1
        log "[Group $k] Inference complete at $(date +%H:%M:%S)."

        # Launch merge + cache cleanup in background — don't block slot from being freed
        (
            # Merge predictions to final .las files (CPU only, no GPU needed)
            NPZ_COUNT=$(find "$group_output" -maxdepth 1 -name "predictions_*.npz" 2>/dev/null | wc -l)
            if [ -f "$group_output/eval.yaml" ] && [ "$NPZ_COUNT" -gt 0 ]; then
                log "[Group $k] Starting merge at $(date +%H:%M:%S)..."
                python3 "$SCRIPT_DIR/nibio_inference/merge_predictions.py" \
                    -e "$group_output/eval.yaml" \
                    -p "$group_output" \
                    -o "$final_dir" \
                    -v >> "$group_log" 2>&1
                log "[Group $k] Merge complete at $(date +%H:%M:%S)."
            elif [ "$NPZ_COUNT" -eq 0 ]; then
                log "[Group $k] No predictions produced (no instances detected), skipping merge."
            fi
            # Clean cached .pt files
            if [ -d "$CACHE_DIR" ]; then
                for ply in "$group_input"/*.ply; do
                    [ -e "$ply" ] || continue
                    base=$(basename "$ply" .ply)
                    rm -f "$CACHE_DIR/processed_${base}.pt" "$CACHE_DIR/raw_area_${base}.pt"
                done
            fi
        ) &
    ) &
    SLOT_PIDS[$slot]=$!
    SLOT_GPU[$slot]=$gpu
    SLOT_GROUP[$slot]=$k
    log "Launched group $k on GPU $gpu slot $slot (PID $!)"
}

# Slot-based worker pool: each slot = one concurrent worker
# Slots 0..MAX_CONCURRENT-1, mapped round-robin to GPUs
declare -A SLOT_PIDS  # slot -> PID
declare -A SLOT_GPU   # slot -> GPU id
declare -A SLOT_GROUP # slot -> group number
FAILED=0
NEXT_GROUP=0

# Launch initial batch: fill all slots
for ((slot=0; slot<MAX_CONCURRENT && NEXT_GROUP<${#VALID_GROUPS[@]}; slot++)); do
    gpu=$((slot % N_GPUS))
    k=${VALID_GROUPS[$NEXT_GROUP]}
    launch_group "$k" "$gpu" "$slot"
    NEXT_GROUP=$((NEXT_GROUP + 1))
done

# Poll every 5s — when a slot frees up, launch the next group on the same GPU
while [ $NEXT_GROUP -lt ${#VALID_GROUPS[@]} ] || [ ${#SLOT_PIDS[@]} -gt 0 ]; do
    sleep 5
    for slot in "${!SLOT_PIDS[@]}"; do
        pid=${SLOT_PIDS[$slot]}
        if [ ! -d "/proc/$pid" ]; then
            wait "$pid" 2>/dev/null || true
            EXIT_CODE=$?
            gpu=${SLOT_GPU[$slot]}
            grp=${SLOT_GROUP[$slot]}
            if [ "$EXIT_CODE" -ne 0 ]; then
                log "[Group $grp] FAILED on GPU $gpu (exit $EXIT_CODE)"
                FAILED=$((FAILED + 1))
            else
                log "[Group $grp] Inference complete on GPU $gpu at $(date +%H:%M:%S)"
            fi
            unset 'SLOT_PIDS[$slot]'
            unset 'SLOT_GPU[$slot]'
            unset 'SLOT_GROUP[$slot]'

            # Launch next group in the freed slot (same GPU)
            if [ $NEXT_GROUP -lt ${#VALID_GROUPS[@]} ]; then
                k=${VALID_GROUPS[$NEXT_GROUP]}
                launch_group "$k" "$gpu" "$slot"
                NEXT_GROUP=$((NEXT_GROUP + 1))
            fi
        fi
    done
done

if [ "$FAILED" -gt 0 ]; then
    log "ERROR: $FAILED group(s) failed. Check logs above."
    exit 1
fi

log "All groups completed successfully."
log_phase_time "Phase 3 (Inference total)" $PHASE_START

# ============================================================
# PHASE 4: Collect results and merge
# ============================================================
log ""
log "=== Phase 4: Collecting results and merging ==="
PHASE_START=$SECONDS

FINAL_DEST_DIR="$DEST_DIR/final_results"
mkdir -p "$FINAL_DEST_DIR"

# Merging happens inline after each group's inference (launched in background).
# Wait for ALL merge processes to finish (including grandchild processes).
log "Waiting for background merge processes to complete..."
wait
# Also wait for any merge_predictions.py processes still running
while pgrep -f "merge_predictions.py.*$DEST_DIR" > /dev/null 2>&1; do
    log "Merge processes still running, waiting..."
    sleep 10
done

# Check for any groups that may have failed merging and retry
for ((k=0; k<N_GPU_WORKERS; k++)); do
    GROUP_OUTPUT_DIR="$DEST_DIR/group_${k}"
    if [ -f "$GROUP_OUTPUT_DIR/eval.yaml" ]; then
        # Skip groups that produced no prediction files (no instances detected)
        NPZ_COUNT=$(find "$GROUP_OUTPUT_DIR" -maxdepth 1 -name "predictions_*.npz" 2>/dev/null | wc -l)
        if [ "$NPZ_COUNT" -eq 0 ]; then
            log "Group $k: No prediction files (no instances detected), skipping merge."
            continue
        fi

        # Check if this group produced any .las output
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
            log "Group $k: $MISSING files missing, re-running merge..."
            python3 "$SCRIPT_DIR/nibio_inference/merge_predictions.py" \
                -e "$GROUP_OUTPUT_DIR/eval.yaml" \
                -p "$GROUP_OUTPUT_DIR" \
                -o "$FINAL_DEST_DIR" \
                -v
        fi
    fi
done

log_phase_time "Phase 4 (Merging)" $PHASE_START

num_files=$(find "$FINAL_DEST_DIR" -maxdepth 1 -type f | wc -l)

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

log ""
log "=== Complete ==="
log "Finished at: $(date)"
log "[TIMING] Total elapsed: ${TOTAL_MINS}m ${TOTAL_SECS}s"
log "Number of files in final results: $num_files"
log "Results directory: $FINAL_DEST_DIR"
log "Log file: $LOG_FILE"
