"""Merge .npz prediction files with original point clouds into .las files.

Replaces the PLY-based merge pipeline (merge_pt_ss_is.py + rename scripts)
with direct index-based merging using KDTree — no coordinate string matching.
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.pandas_to_las import pandas_to_las


def merge_single(fold_path, predictions_path, output_path, verbose=False):
    """Merge a single .npz prediction file with its original point cloud."""
    if verbose:
        print(f"Merging: {os.path.basename(fold_path)}")

    # Load predictions (subsampled positions + labels)
    preds = np.load(predictions_path)
    pred_pos = preds['pos']
    semantic_labels = preds['semantic_labels']
    instance_labels = preds['instance_labels']

    # Load original point cloud from utm2local
    original_df = ply_to_pandas(fold_path)
    original_df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)

    # KDTree match: for each original point, find nearest prediction point
    tree = cKDTree(pred_pos)
    distances, indices = tree.query(original_df[['x', 'y', 'z']].values, k=1)

    # Assign predictions
    original_df['PredSemantic'] = semantic_labels[indices]
    original_df['PredInstance'] = instance_labels[indices].astype(np.float64)

    # If distance > 1m, mark as no instance (same threshold as tracker)
    original_df.loc[distances > 1.0, 'PredInstance'] = -1

    # Add 1 to PredInstance (match existing merge behavior)
    original_df['PredInstance'] = original_df['PredInstance'] + 1

    # Assign NaN-equivalent (0) for points without instance
    original_df['PredInstance'] = original_df['PredInstance'].clip(lower=0)

    # Add back UTM coordinates
    min_values_path = fold_path.replace('.ply', '_min_values.json')
    with open(min_values_path, 'r') as f:
        min_values = json.load(f)

    min_x, min_y, min_z = min_values
    original_df['x'] = original_df['x'].astype(float) + min_x
    original_df['y'] = original_df['y'].astype(float) + min_y
    original_df['z'] = original_df['z'].astype(float) + min_z

    # Write .las
    pandas_to_las(
        original_df,
        csv_file_provided=False,
        output_file_path=output_path,
        do_compress=True,
        verbose=verbose
    )

    if verbose:
        print(f"Written: {output_path}")


def merge_all(eval_yaml_path, predictions_dir, output_dir, verbose=False):
    """Merge all prediction files referenced by an eval.yaml."""
    with open(eval_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    fold = config.get('data', {}).get('fold', [])
    if not fold:
        print(f"ERROR: No fold entries in {eval_yaml_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for i, fold_path in enumerate(fold):
        predictions_path = os.path.join(predictions_dir, f"predictions_{i}.npz")
        if not os.path.exists(predictions_path):
            print(f"WARNING: Missing {predictions_path}, skipping")
            continue

        # Output filename: strip number prefix and use original name
        base_name = os.path.splitext(os.path.basename(fold_path))[0]
        # Remove trailing _out suffix if present (from utm2local)
        if base_name.endswith('_out'):
            base_name = base_name[:-4]
        output_path = os.path.join(output_dir, f"{base_name}.las")
        tasks.append((fold_path, predictions_path, output_path))

    if not tasks:
        print(f"No prediction files found in {predictions_dir}, nothing to merge.")
        return

    if verbose:
        print(f"Merging {len(tasks)} files...")

    n_jobs = min(len(tasks), int(os.environ.get('MERGE_JOBS', 4)))
    Parallel(n_jobs=n_jobs)(
        delayed(merge_single)(fold_path, pred_path, out_path, verbose)
        for fold_path, pred_path, out_path in tasks
    )

    print(f"Merge complete. {len(tasks)} files written to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge .npz predictions with original point clouds into .las files.'
    )
    parser.add_argument('-e', '--eval_yaml', type=str, required=True,
                        help='Path to eval.yaml (contains fold list)')
    parser.add_argument('-p', '--predictions_dir', type=str, required=True,
                        help='Directory containing predictions_*.npz files')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Output directory for .las files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose output')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    merge_all(args.eval_yaml, args.predictions_dir, args.output_dir, args.verbose)
