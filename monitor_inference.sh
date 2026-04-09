#!/bin/bash
# Monitor script for parallel inference progress
# Usage: ./monitor_inference.sh [results_dir]
# Auto-refreshes every 30s with: watch -n30 ./monitor_inference.sh

DEST_DIR="${1:-/home/tmp_results}"
LOG_FILE=$(ls -t "$DEST_DIR"/inference_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No inference log found in $DEST_DIR"
    exit 1
fi

# Get config from log
TOTAL_GROUPS=$(grep "Total groups:" "$LOG_FILE" | grep -oP "\d+")
N_GPUS=$(grep "GPUs available:" "$LOG_FILE" | grep -oP "\d+")
WORKERS_PER_GPU=$(grep "Workers per GPU:" "$LOG_FILE" | grep -oP "^\d+" | head -1)
MAX_CONCURRENT=$(grep -oP "max concurrent: \K\d+" "$LOG_FILE")

echo "============================================"
echo "  Inference Monitor  —  $(date '+%H:%M:%S')"
echo "============================================"
echo "  Config: $TOTAL_GROUPS groups | $N_GPUS GPUs | $MAX_CONCURRENT concurrent"

# GPU status
echo ""
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | while read line; do
    echo "  GPU $line"
done

# Memory & Disk
echo ""
echo "--- Resources ---"
MEM=$(free -h | awk '/Mem:/{printf "RAM: %s / %s (avail %s)", $3, $2, $7}')
DISK=$(df -h /home | awk 'NR==2{printf "Disk: %s / %s (%s)", $3, $2, $5}')
echo "  $MEM"
echo "  $DISK"

# Completed groups — only count as done when merge has finished,
# or when inference produced no predictions (no merge needed)
mapfile -t DONE_LINES < <(grep -E "\] Merge complete|\] No predictions produced" "$LOG_FILE")
COMPLETED_COUNT=${#DONE_LINES[@]}

# Identify main eval.py processes: parent is the shell script, not another python
# Use PPID — main processes have the subshell as parent, dataloader workers have eval.py as parent
declare -A MAIN_PROCS
while IFS= read -r line; do
    PID=$(echo "$line" | awk '{print $1}')
    PPID_VAL=$(echo "$line" | awk '{print $2}')
    CPU=$(echo "$line" | awk '{print $3}')
    CONFIG=$(echo "$line" | grep -oP "group_\d+")
    # Check if parent is NOT python (i.e., this is a main process, not a dataloader worker)
    PARENT_CMD=$(ps -p "$PPID_VAL" -o comm= 2>/dev/null)
    if [[ "$PARENT_CMD" != python* ]]; then
        MAIN_PROCS[$CONFIG]="$PID|$CPU"
    fi
done < <(ps -eo pid,ppid,%cpu,args | grep "eval.py --config-name" | grep -v grep)

RUNNING_COUNT=${#MAIN_PROCS[@]}

echo ""
echo "--- Progress: $COMPLETED_COUNT / $TOTAL_GROUPS complete | $(grep -c "\] Starting merge" "$LOG_FILE" 2>/dev/null || echo 0) merging | $RUNNING_COUNT inferring ---"

# Completed groups
if [ "$COMPLETED_COUNT" -gt 0 ]; then
    echo ""
    echo "Completed:"
    for line in "${DONE_LINES[@]}"; do
        GRP=$(echo "$line" | grep -oP "Group \d+")
        TIME=$(echo "$line" | grep -oP "at \S+")
        echo "  done  $GRP (merge $TIME)"
    done
fi

# Merging groups: inference done, merge started but not yet complete
MERGING_GROUPS=()
for ((g=0; g<TOTAL_GROUPS; g++)); do
    if grep -q "\[Group $g\] Starting merge" "$LOG_FILE" 2>/dev/null; then
        if ! grep -qE "\[Group $g\] Merge complete|\[Group $g\] No predictions produced" "$LOG_FILE" 2>/dev/null; then
            MERGE_START=$(grep "\[Group $g\] Starting merge" "$LOG_FILE" | grep -oP "at \S+" | tail -1)
            LAS_COUNT=$(ls "$DEST_DIR/final_results/"*.las "$DEST_DIR/final_results/"*.laz 2>/dev/null | wc -l)
            MERGING_GROUPS+=("$g|$MERGE_START")
        fi
    fi
done

if [ ${#MERGING_GROUPS[@]} -gt 0 ]; then
    echo ""
    echo "Merging (${#MERGING_GROUPS[@]}):"
    for entry in "${MERGING_GROUPS[@]}"; do
        IFS='|' read -r GRP_NUM MERGE_START <<< "$entry"
        NPZ_COUNT=$(ls "$DEST_DIR/group_${GRP_NUM}"/predictions_*.npz 2>/dev/null | wc -l)
        TILES_TOTAL=$(ls "$DEST_DIR/group_${GRP_NUM}_input"/*.ply 2>/dev/null | wc -l)
        MERGE_PROC=""
        if pgrep -f "merge_predictions.py.*group_${GRP_NUM}" > /dev/null 2>&1; then
            MERGE_PROC="  [merge running]"
        fi
        echo "  merge  Group $GRP_NUM  (started $MERGE_START  npz:$NPZ_COUNT/$TILES_TOTAL)$MERGE_PROC"
    done
fi

# Active workers
echo ""
echo "Active workers ($RUNNING_COUNT):"
for CONFIG in $(echo "${!MAIN_PROCS[@]}" | tr ' ' '\n' | sort -t_ -k2 -n); do
    IFS='|' read -r PID CPU <<< "${MAIN_PROCS[$CONFIG]}"
    TILES_DONE=$(ls "$DEST_DIR/${CONFIG}"/predictions_*.npz 2>/dev/null | wc -l)
    TILES_TOTAL=$(ls "$DEST_DIR/${CONFIG}_input"/*.ply 2>/dev/null | wc -l)

    # Get group number to figure out GPU assignment
    GRP_NUM=$(echo "$CONFIG" | grep -oP "\d+")
    GPU_ID=$((GRP_NUM % N_GPUS))

    # Get progress from group's worker.log
    WORKER_LOG="$DEST_DIR/${CONFIG}/worker.log"
    PROGRESS_PCT=""
    if [ -f "$WORKER_LOG" ]; then
        PROGRESS_PCT=$(grep -oP "\d+%" "$WORKER_LOG" | tail -1)
    fi
    PROGRESS_STR="${PROGRESS_PCT:- starting}"

    echo "  run   $CONFIG  GPU:$GPU_ID  PID:$PID  CPU:${CPU}%  tiles:$TILES_DONE/$TILES_TOTAL  $PROGRESS_STR"
done

# Queued groups
LAUNCHED_COUNT=$(grep -oP "Launched group \K\d+" "$LOG_FILE" | sort -n | uniq | wc -l)
QUEUED=$((TOTAL_GROUPS - LAUNCHED_COUNT))
echo ""
echo "Queued: $QUEUED groups waiting"

# Time estimate
if [ "$COMPLETED_COUNT" -gt 0 ]; then
    START_DATE=$(grep "Started at:" "$LOG_FILE" | sed 's/Started at: //')
    START_EPOCH=$(date -d "$START_DATE" +%s 2>/dev/null)
    NOW_EPOCH=$(date +%s)

    if [ -n "$START_EPOCH" ]; then
        ELAPSED=$((NOW_EPOCH - START_EPOCH))
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_H=$((ELAPSED_MIN / 60))
        ELAPSED_M=$((ELAPSED_MIN % 60))

        # Throughput: completed groups per second of wall time, scaled by concurrency
        # Each completed group used ~1/MAX_CONCURRENT of wall time
        # So effective per-group time = elapsed * MAX_CONCURRENT / completed
        # Remaining wall time = (remaining_groups * per_group_time) / MAX_CONCURRENT
        # Simplifies to: remaining_wall = elapsed * remaining / completed
        REMAINING_GROUPS=$((TOTAL_GROUPS - COMPLETED_COUNT))
        EST_REMAINING=$((ELAPSED * REMAINING_GROUPS / COMPLETED_COUNT))
        EST_H=$((EST_REMAINING / 3600))
        EST_M=$(( (EST_REMAINING % 3600) / 60))
        FINISH_EPOCH=$((NOW_EPOCH + EST_REMAINING))
        FINISH_TIME=$(date -d "@$FINISH_EPOCH" '+%H:%M' 2>/dev/null)
        TOTAL_EST=$((ELAPSED + EST_REMAINING))
        TOTAL_H=$((TOTAL_EST / 3600))
        TOTAL_M=$(( (TOTAL_EST % 3600) / 60))

        echo ""
        echo "--- Time Estimate ---"
        echo "  Elapsed:     ${ELAPSED_H}h ${ELAPSED_M}m"
        echo "  Remaining:   ~${EST_H}h ${EST_M}m  ($REMAINING_GROUPS groups left)"
        echo "  Total est:   ~${TOTAL_H}h ${TOTAL_M}m"
        echo "  ETA:         ~$FINISH_TIME"
    fi
fi

echo ""
