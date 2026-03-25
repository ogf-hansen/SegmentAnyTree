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


def write_las_to_s3(las_data, s3_uri):
    """Write a LasData object directly to S3 without local file."""
    import io
    import boto3
    # Parse s3://bucket/key
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    buf = io.BytesIO()
    las_data.write(buf)
    buf.seek(0)
    boto3.client('s3').upload_fileobj(buf, bucket, key)


def retile_single(tile_path, neighbor_paths, overlap, output_path,
                   s3_uri=None, write=False, verbose=False):
    """Read tile + neighbor border strips, concatenate, write .las."""
    target = laspy.read(tile_path)
    xmin, xmax = target.header.x_min, target.header.x_max
    ymin, ymax = target.header.y_min, target.header.y_max

    border_points = []
    border_headers = []
    for nb_path in neighbor_paths:
        nb = laspy.read(nb_path)
        x = np.array(nb.x)
        y = np.array(nb.y)
        # Points inside expanded bbox
        mask = (
            (x >= xmin - overlap) & (x <= xmax + overlap) &
            (y >= ymin - overlap) & (y <= ymax + overlap)
        )
        # Exclude points already inside original bbox (avoid duplication)
        inner = (
            (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        )
        strip_mask = mask & ~inner
        if np.any(strip_mask):
            border_points.append(nb.points[strip_mask])

    if border_points:
        # Re-encode border points using the target tile's scale/offset.
        # Each LAS tile stores coordinates as scaled integers:
        #   utm_coord = raw_int * scale + offset
        # Tiles may have different offsets, so raw integers from a neighbor
        # decoded with the target's offset would produce wrong coordinates.
        target_scales = target.header.scales
        target_offsets = target.header.offsets

        re_encoded_arrays = []
        for bp in border_points:
            # Decode via laspy properties (applies neighbor's scale/offset automatically)
            bp_x = bp.x
            bp_y = bp.y
            bp_z = bp.z
            # Re-encode using target's scale/offset into a fresh copy
            arr = bp.array.copy()
            arr['X'] = np.round((bp_x - target_offsets[0]) / target_scales[0]).astype(np.int32)
            arr['Y'] = np.round((bp_y - target_offsets[1]) / target_scales[1]).astype(np.int32)
            arr['Z'] = np.round((bp_z - target_offsets[2]) / target_scales[2]).astype(np.int32)
            re_encoded_arrays.append(arr)

        all_arrays = [target.points.array] + re_encoded_arrays
        merged_array = np.concatenate(all_arrays)
        output = laspy.LasData(target.header)
        output.points = laspy.ScaleAwarePointRecord(
            merged_array, target.header.point_format,
            scales=target.header.scales, offsets=target.header.offsets
        )
        if write:
            if s3_uri:
                s3_key = s3_uri.rstrip("/") + "/" + os.path.basename(output_path)
                write_las_to_s3(output, s3_key)
            else:
                output.write(output_path)
        if verbose:
            added = len(merged_array) - len(target.points)
            print(f"{os.path.basename(tile_path)}: {len(target.points)} + {added} overlap = {len(merged_array)} points")
    else:
        if write:
            if s3_uri:
                s3_key = s3_uri.rstrip("/") + "/" + os.path.basename(output_path)
                write_las_to_s3(target, s3_key)
            else:
                target.write(output_path)
        if verbose:
            print(f"{os.path.basename(tile_path)}: no neighbors in range, copied as-is")


def main(input_dir, output_dir, overlap=5.0, n_jobs=-1, s3_uri=None, write=False, verbose=False):
    """Build spatial index from headers, then retile in parallel."""
    if output_dir:
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
        basename = os.path.basename(path)
        if basename.lower().endswith('.laz'):
            basename = basename[:-4] + '.las'
        output_path = os.path.join(output_dir, basename) if output_dir else basename
        tasks.append((path, neighbors, output_path))
        # Store by basename for portability
        tile_name = os.path.splitext(os.path.basename(path))[0]
        neighbors_index[tile_name] = [
            os.path.splitext(os.path.basename(nb))[0] for nb in neighbors
        ]

    # Write neighbor index JSON
    if output_dir:
        neighbors_json_path = os.path.join(output_dir, "neighbors.json")
        with open(neighbors_json_path, 'w') as f:
            json.dump(neighbors_index, f, indent=2)
        if verbose:
            print(f"Neighbor index written to {neighbors_json_path}")

    print(f"Retiling {len(tasks)} tiles with {overlap}m overlap (n_jobs={n_jobs})...")
    Parallel(n_jobs=n_jobs)(
        delayed(retile_single)(path, neighbors, overlap, out, s3_uri, write, verbose)
        for path, neighbors, out in tqdm(tasks, desc="Retiling", disable=not verbose)
    )
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Retile point cloud tiles with overlap by borrowing border points from neighbors."
    )
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help="Directory containing input .las/.laz tiles")
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help="Directory to write retiled .las files (required unless --s3_uri is set)")
    parser.add_argument('--overlap', type=float, default=5.0,
                        help="Overlap distance in meters (default: 5.0)")
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help="Number of parallel jobs (default: -1 = all cores)")
    parser.add_argument('--s3_uri', type=str, default=None,
                        help="Optional S3 URI to upload output tiles (e.g. s3://bucket/folder)")
    parser.add_argument('--write', action='store_true',
                        help="Actually write output files (default: dry run)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print per-tile statistics")

    args = parser.parse_args()
    if args.write and not args.s3_uri and not args.output_dir:
        parser.error("--output_dir is required when writing locally (without --s3_uri)")
    main(args.input_dir, args.output_dir, args.overlap, args.n_jobs,
         args.s3_uri, args.write, args.verbose)
