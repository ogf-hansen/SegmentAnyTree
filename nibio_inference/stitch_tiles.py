"""Stitch segmented tiles back together after overlap-based inference.

After retile_with_overlap.py adds 5m border points and inference runs on
the expanded tiles, this script:
1. Uses original tile headers to separate core vs overlap points
2. Matches instances across tiles in the overlap zones using IoU
3. Assigns globally consistent instance IDs via union-find
4. Outputs clean tiles (core points only) with stitched instance labels

Usage:
    python nibio_inference/stitch_tiles.py \
        --original_dir /path/to/original_tiles/ \
        --segmented_dir /path/to/segmented_tiles/ \
        --output_dir /path/to/stitched_output/ \
        --overlap 5.0 \
        --iou_threshold 0.3 \
        -v
"""

import argparse
import json
import os
import sys

import laspy
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Union-Find for merging instance IDs across tiles
# ---------------------------------------------------------------------------

class UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_tile_bounds(tile_path):
    """Read laspy header only, return (xmin, xmax, ymin, ymax)."""
    with laspy.open(tile_path) as f:
        h = f.header
        return (h.x_min, h.x_max, h.y_min, h.y_max)


def classify_points(x, y, core_bounds):
    """Return boolean mask: True for core points, False for overlap points."""
    xmin, xmax, ymin, ymax = core_bounds
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def match_instances_between_tiles(
    overlap_xyz, overlap_instances,
    neighbor_xyz, neighbor_instances,
    iou_threshold=0.3, match_radius=0.05,
):
    """Find instance correspondences in the overlap zone.

    Parameters
    ----------
    overlap_xyz : (N, 3) array
        Positions of overlap points from tile A (borrowed from B's territory).
    overlap_instances : (N,) array
        Instance labels assigned by tile A's segmentation.
    neighbor_xyz : (M, 3) array
        Positions of core points from tile B in the same spatial region.
    neighbor_instances : (M,) array
        Instance labels assigned by tile B's segmentation.
    iou_threshold : float
        Minimum IoU to consider two instances as the same object.
    match_radius : float
        Maximum distance (m) to consider two points as the same physical point.

    Returns
    -------
    list of (instance_a, instance_b) pairs that should be merged.
    """
    if len(overlap_xyz) == 0 or len(neighbor_xyz) == 0:
        return []

    # Find corresponding points between overlap and neighbor core
    tree = cKDTree(neighbor_xyz)
    distances, indices = tree.query(overlap_xyz, k=1)
    matched = distances < match_radius

    if not np.any(matched):
        return []

    # For matched points, get instance labels from both sides
    a_inst = overlap_instances[matched]
    b_inst = neighbor_instances[indices[matched]]

    # Skip background (instance 0)
    valid = (a_inst > 0) & (b_inst > 0)
    if not np.any(valid):
        return []

    a_inst = a_inst[valid]
    b_inst = b_inst[valid]

    # Build sets for IoU computation
    # For each instance in A, collect the set of matched-point indices
    a_labels = np.unique(a_inst)
    b_labels = np.unique(b_inst)

    a_sets = {int(l): set(np.where(a_inst == l)[0].tolist()) for l in a_labels}
    b_sets = {int(l): set(np.where(b_inst == l)[0].tolist()) for l in b_labels}

    merges = []
    for al, a_set in a_sets.items():
        best_iou = 0.0
        best_bl = None
        for bl, b_set in b_sets.items():
            inter = len(a_set & b_set)
            union = len(a_set | b_set)
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_bl = bl
        if best_iou >= iou_threshold and best_bl is not None:
            merges.append((al, best_bl))

    return merges


def stitch_tiles(
    original_dir,
    segmented_dir,
    output_dir,
    neighbors_json,
    overlap=5.0,
    iou_threshold=0.3,
    verbose=False,
):
    """Main stitching pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Load neighbor index from retile_with_overlap ------------------
    with open(neighbors_json, 'r') as f:
        neighbors_index = json.load(f)

    if verbose:
        print(f"Loaded neighbor index: {len(neighbors_index)} tiles")

    # ---- 2. Discover tiles ------------------------------------------------
    original_files = {}
    for f in os.listdir(original_dir):
        if f.lower().endswith(('.las', '.laz')):
            original_files[os.path.splitext(f)[0]] = os.path.join(original_dir, f)

    segmented_files = {}
    for f in os.listdir(segmented_dir):
        if f.lower().endswith(('.las', '.laz')):
            segmented_files[os.path.splitext(f)[0]] = os.path.join(segmented_dir, f)

    common = sorted(
        set(original_files.keys()) & set(segmented_files.keys()) & set(neighbors_index.keys())
    )
    if not common:
        print("ERROR: No matching tile names across original, segmented, and neighbors.json.")
        sys.exit(1)

    if verbose:
        print(f"Found {len(common)} matching tiles")

    # ---- 3. Read original headers for core bounds -------------------------
    core_bounds = {}
    for name in tqdm(common, desc="Reading original headers", disable=not verbose):
        core_bounds[name] = get_tile_bounds(original_files[name])

    # ---- 4. Load segmented tiles, separate core/overlap -------------------
    tile_data = {}
    for name in tqdm(common, desc="Loading segmented tiles", disable=not verbose):
        las = laspy.read(segmented_files[name])
        x = np.array(las.x)
        y = np.array(las.y)
        z = np.array(las.z)

        instances = np.array(las['PredInstance'])
        semantics = np.array(las['PredSemantic'])

        core_mask = classify_points(x, y, core_bounds[name])

        tile_data[name] = {
            'las': las,
            'x': x, 'y': y, 'z': z,
            'instances': instances,
            'semantics': semantics,
            'core_mask': core_mask,
        }

    # ---- 5. Match instances across neighbor pairs -------------------------
    uf = UnionFind()
    names_list = list(common)
    processed_pairs = set()

    n_merges_total = 0
    for name_a in tqdm(names_list, desc="Matching instances", disable=not verbose):
        da = tile_data[name_a]
        ax_min, ax_max, ay_min, ay_max = core_bounds[name_a]

        overlap_mask_a = ~da['core_mask']
        if not np.any(overlap_mask_a):
            continue

        overlap_x = da['x'][overlap_mask_a]
        overlap_y = da['y'][overlap_mask_a]
        overlap_z = da['z'][overlap_mask_a]
        overlap_inst = da['instances'][overlap_mask_a]

        for name_b in neighbors_index.get(name_a, []):
            if name_b not in tile_data:
                continue

            # Only process each pair once
            pair = tuple(sorted([name_a, name_b]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            db = tile_data[name_b]
            bx_min, bx_max, by_min, by_max = core_bounds[name_b]

            # B's core points
            b_core = db['core_mask']
            bx = db['x'][b_core]
            by = db['y'][b_core]
            bz = db['z'][b_core]
            b_inst = db['instances'][b_core]

            # Filter A's overlap points to those inside B's core bounds
            in_b = (
                (overlap_x >= bx_min) & (overlap_x <= bx_max) &
                (overlap_y >= by_min) & (overlap_y <= by_max)
            )
            if not np.any(in_b):
                continue

            # Filter B's core to the overlap region with A
            in_a_expanded = (
                (bx >= ax_min - overlap) & (bx <= ax_max + overlap) &
                (by >= ay_min - overlap) & (by <= ay_max + overlap)
            )
            if not np.any(in_a_expanded):
                continue

            overlap_xyz = np.column_stack(
                [overlap_x[in_b], overlap_y[in_b], overlap_z[in_b]]
            )
            overlap_instances = overlap_inst[in_b]

            neighbor_xyz = np.column_stack(
                [bx[in_a_expanded], by[in_a_expanded], bz[in_a_expanded]]
            )
            neighbor_instances = b_inst[in_a_expanded]

            merges = match_instances_between_tiles(
                overlap_xyz, overlap_instances,
                neighbor_xyz, neighbor_instances,
                iou_threshold=iou_threshold,
            )

            for inst_a, inst_b in merges:
                key_a = (name_a, int(inst_a))
                key_b = (name_b, int(inst_b))
                uf.union(key_a, key_b)
                n_merges_total += 1

    if verbose:
        print(f"Total instance merges across tiles: {n_merges_total}")

    # ---- 5. Build global instance ID mapping ------------------------------
    # Collect all (tile, instance) keys that participate in merges
    # For each tile, map local instance → global instance
    global_counter = 1
    root_to_global = {}

    tile_mappings = {name: {} for name in names_list}

    for name in tqdm(names_list):
        da = tile_data[name]
        local_instances = np.unique(da['instances'][da['core_mask']])
        for inst in local_instances:
            inst = int(inst)
            if inst == 0:
                continue  # background stays 0
            key = (name, inst)
            root = uf.find(key)
            if root not in root_to_global:
                root_to_global[root] = global_counter
                global_counter += 1
            tile_mappings[name][inst] = root_to_global[root]

    if verbose:
        print(f"Global instance count: {global_counter - 1}")

    # ---- 6. Write output tiles (core points only, remapped instances) -----
    for name in tqdm(names_list, desc="Writing output", disable=not verbose):
        da = tile_data[name]
        core = da['core_mask']
        las_in = da['las']

        # Extract core points
        core_points = las_in.points[core]

        # Remap instance IDs
        instances = np.array(da['instances'][core])
        mapping = tile_mappings[name]
        remapped = np.zeros(len(instances), dtype=np.uint32)
        for local_id, global_id in mapping.items():
            remapped[instances == local_id] = global_id

        # Create output LAS with core points only, PredInstance as uint32
        output = laspy.LasData(las_in.header)
        output.points = core_points

        # Replace PredInstance with uint32 version
        if 'PredInstance' in output.point_format.dimension_names:
            output.remove_extra_dim('PredInstance')
        output.add_extra_dim(laspy.ExtraBytesParams(
            name='PredInstance', type=np.uint32,
            description="Remapped instance ID"
        ))
        output['PredInstance'] = remapped

        # Copy VLRs (contains CRS/projection info)
        output.vlrs = list(las_in.vlrs)

        # Copy EVLRs (LAS 1.4 extended variable length records)
        if hasattr(las_in, 'evlrs') and las_in.evlrs:
            output.evlrs = list(las_in.evlrs)


        # Write
        out_name = os.path.basename(segmented_files[name])
        # Ensure .laz extension
        if out_name.lower().endswith('.las'):
            out_name = out_name[:-4] + '.laz'
        output_path = os.path.join(output_dir, out_name)
        output.write(output_path)

        if verbose:
            n_core = int(core.sum())
            n_total = len(da['x'])
            print(f"{name}: {n_core}/{n_total} core points, "
                  f"{len(mapping)} instances remapped")

    print(f"Stitching complete. {len(names_list)} tiles written to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Stitch segmented tiles: resolve instance IDs across tile boundaries."
    )
    parser.add_argument('--original_dir', type=str, required=True,
                        help="Directory of original tiles (pre-overlap, for core bounds)")
    parser.add_argument('--segmented_dir', type=str, required=True,
                        help="Directory of segmented tiles (post-inference, with PredInstance/PredSemantic)")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory for stitched tiles")
    parser.add_argument('--neighbors_json', type=str, required=True,
                        help="Path to neighbors.json from retile_with_overlap.py")
    parser.add_argument('--overlap', type=float, default=5.0,
                        help="Overlap distance used during retiling (default: 5.0)")
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help="Minimum IoU to merge instances across tiles (default: 0.3)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print per-tile statistics")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    stitch_tiles(
        args.original_dir,
        args.segmented_dir,
        args.output_dir,
        args.neighbors_json,
        overlap=args.overlap,
        iou_threshold=args.iou_threshold,
        verbose=args.verbose,
    )
