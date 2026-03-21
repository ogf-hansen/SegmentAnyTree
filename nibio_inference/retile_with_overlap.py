import json
import os
import argparse

import numpy as np
import laspy
from joblib import Parallel, delayed
from tqdm import tqdm


def get_tile_bounds(tile_path):
    """Read laspy header only, return (xmin, xmax, ymin, ymax)."""
    with laspy.open(tile_path) as f:
        h = f.header
        return (h.x_min, h.x_max, h.y_min, h.y_max)


def find_neighbors(target_path, target_bounds, all_bounds, overlap):
    """Return list of tile paths whose bbox intersects target expanded by overlap."""
    xmin, xmax, ymin, ymax = target_bounds
    exp_xmin = xmin - overlap
    exp_xmax = xmax + overlap
    exp_ymin = ymin - overlap
    exp_ymax = ymax + overlap

    neighbors = []
    for path, (bxmin, bxmax, bymin, bymax) in all_bounds.items():
        if path == target_path:
            continue
        # Check bbox overlap
        if bxmax >= exp_xmin and bxmin <= exp_xmax and bymax >= exp_ymin and bymin <= exp_ymax:
            neighbors.append(path)
    return neighbors


def upload_dir_to_s3(local_dir, s3_uri, verbose=False):
    """Upload all files in a local directory to S3 using the AWS CLI."""
    import subprocess
    s3_dest = s3_uri.rstrip("/") + "/"
    cmd = ["aws", "s3", "sync", local_dir, s3_dest]
    if verbose:
        print(f"Uploading {local_dir} → {s3_dest}")
    subprocess.run(cmd, check=True)


def retile_single(tile_path, neighbor_paths, overlap, output_path,
                   verbose=False):
    """Read tile + neighbor border strips, concatenate, write .las."""
    target = laspy.read(tile_path)
    xmin, xmax = target.header.x_min, target.header.x_max
    ymin, ymax = target.header.y_min, target.header.y_max

    border_points = []
    for nb_path in neighbor_paths:
        nb = laspy.read(nb_path)
        x = nb.x
        y = nb.y
        # Points inside expanded bbox
        mask = (
            (x >= xmin - overlap) & (x <= xmax + overlap) &
            (y >= ymin - overlap) & (y <= ymax + overlap)
        )
        # Exclude points already inside original bbox (avoid duplication)
        inner = (
            (x >= xmin) & (x <= xmax) &
            (y >= ymin) & (y <= ymax)
        )
        strip_mask = mask & ~inner
        if np.any(strip_mask):
            border_points.append(nb.points[strip_mask])

    if border_points:
        all_arrays = [target.points.array] + [bp.array for bp in border_points]
        merged_array = np.concatenate(all_arrays)
        output = laspy.LasData(target.header)
        output.points = laspy.ScaleAwarePointRecord(
            merged_array, target.header.point_format,
            scales=target.header.scales, offsets=target.header.offsets
        )
        output.write(output_path)
        if verbose:
            added = len(merged_array) - len(target.points)
            print(f"{os.path.basename(tile_path)}: {len(target.points)} + {added} overlap = {len(merged_array)} points")
    else:
        target.write(output_path)
        if verbose:
            print(f"{os.path.basename(tile_path)}: no neighbors in range, copied as-is")


def main(input_dir, output_dir, overlap=5.0, n_jobs=-1, s3_uri=None, verbose=False):
    """Build spatial index from headers, then retile in parallel."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all .las/.laz files
    tile_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.las', '.laz'))
    ]

    if not tile_files:
        print(f"No .las/.laz files found in {input_dir}")
        return

    print(f"Reading headers from {len(tile_files)} tiles...")
    all_bounds = {}
    for path in tqdm(tile_files, desc="Headers", disable=not verbose):
        all_bounds[path] = get_tile_bounds(path)

    # Precompute neighbors for each tile
    tasks = []
    neighbors_index = {}
    for path in tile_files:
        neighbors = find_neighbors(path, all_bounds[path], all_bounds, overlap)
        output_path = os.path.join(output_dir, os.path.basename(path))
        # Ensure output is .las
        if output_path.lower().endswith('.laz'):
            output_path = output_path[:-4] + '.las'
        tasks.append((path, neighbors, output_path))
        # Store by basename for portability
        tile_name = os.path.splitext(os.path.basename(path))[0]
        neighbors_index[tile_name] = [
            os.path.splitext(os.path.basename(nb))[0] for nb in neighbors
        ]

    # Write neighbor index JSON
    neighbors_json_path = os.path.join(output_dir, "neighbors.json")
    with open(neighbors_json_path, 'w') as f:
        json.dump(neighbors_index, f, indent=2)
    if verbose:
        print(f"Neighbor index written to {neighbors_json_path}")

    print(f"Retiling {len(tasks)} tiles with {overlap}m overlap (n_jobs={n_jobs})...")
    Parallel(n_jobs=n_jobs)(
        delayed(retile_single)(path, neighbors, overlap, out, verbose)
        for path, neighbors, out in tqdm(tasks, desc="Retiling", disable=not verbose)
    )
    print("Done.")
    if s3_uri:
        upload_dir_to_s3(output_dir, s3_uri, verbose)
        print(f"All tiles uploaded to {s3_uri}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Retile point cloud tiles with overlap by borrowing border points from neighbors."
    )
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help="Directory containing input .las/.laz tiles")
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help="Directory to write retiled .las files")
    parser.add_argument('--overlap', type=float, default=5.0,
                        help="Overlap distance in meters (default: 5.0)")
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help="Number of parallel jobs (default: -1 = all cores)")
    parser.add_argument('--s3_uri', type=str, default=None,
                        help="Optional S3 URI to upload output tiles (e.g. s3://bucket/folder)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print per-tile statistics")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.overlap, args.n_jobs,
         args.s3_uri, args.verbose)
