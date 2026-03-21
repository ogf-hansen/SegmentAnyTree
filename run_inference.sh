#!/bin/bash
set -e

# Provide path to the input and output directories and also information if to clean the output directory from command line
SOURCE_DIR="$1"
DEST_DIR="$2"
CLEAN_OUTPUT_DIR="$3"

# Set default values if not provided
: "${SOURCE_DIR:=/home/nibio/mutable-outside-world/data_for_test}"
: "${DEST_DIR:=/home/nibio/mutable-outside-world/data_for_test_results}"
: "${CLEAN_OUTPUT_DIR:=true}"

# If someone fails to provide this correctly then print the usage and exit
if [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ] || [ -z "$CLEAN_OUTPUT_DIR" ]; then
    echo "Usage: run_inference.sh <path_to_input_dir> <path_to_output_dir> <clean_output_dir>"
    exit 1
fi

# Clear the output directory if requested
if [ "$CLEAN_OUTPUT_DIR" = "true" ]; then
    rm -rf "$DEST_DIR"/*
fi

# Check if local paths are provided, change them to absolute paths if so
if [[ "$SOURCE_DIR" != /* ]]; then
    SOURCE_DIR=$(pwd)/"$SOURCE_DIR"
fi

if [[ "$DEST_DIR" != /* ]]; then
    DEST_DIR=$(pwd)/"$DEST_DIR"
fi

# Set the script directory and add it to the PYTHONPATH
SCRIPT_DIR="/home/nibio/mutable-outside-world"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Print the input and output directories
echo "Input directory: $SOURCE_DIR"
echo "Output directory: $DEST_DIR"
echo "Script directory: $SCRIPT_DIR"

# Rename the input files to the format that the inference script expects 
# Change '-' to '_' in the file names

# Copy the input files to the output directory to avoid changing the original files in an input_data directory
mkdir -p "$DEST_DIR/input_data"
cp -r "$SOURCE_DIR/"* "$DEST_DIR/input_data/"

python3 "$SCRIPT_DIR/nibio_inference/fix_naming_of_input_files.py" "$DEST_DIR/input_data"

# UTM normalization 
python3 "$SCRIPT_DIR/nibio_inference/pipeline_utm2local_parallel.py" -i "$DEST_DIR/input_data" -o "$DEST_DIR/utm2local"

# Update the eval.yaml file with the correct paths
cp "$SCRIPT_DIR/conf/eval.yaml" "$DEST_DIR"
python3 "$SCRIPT_DIR/nibio_inference/modify_eval.py" "$DEST_DIR/eval.yaml" "$DEST_DIR/utm2local" "$DEST_DIR"

# clear cache
python3 "$SCRIPT_DIR/nibio_inference/clear_cache.py" --eval_yaml "$DEST_DIR/eval.yaml"

# Run the inference script with the config file
python3 eval.py --config-name "$DEST_DIR/eval.yaml"

echo "Done with inference using the config file: $DEST_DIR/eval.yaml"

FINAL_DEST_DIR="$DEST_DIR/final_results"

# Merge predictions (.npz) with original point clouds into .las files
# This replaces the old rename + PLY-based merge pipeline
python3 "$SCRIPT_DIR/nibio_inference/merge_predictions.py" \
    -e "$DEST_DIR/eval.yaml" \
    -p "$DEST_DIR" \
    -o "$FINAL_DEST_DIR" \
    -v

num_files=$(find "$FINAL_DEST_DIR" -maxdepth 1 -type f | wc -l)

echo "Number of files in the final results directory: $num_files"
