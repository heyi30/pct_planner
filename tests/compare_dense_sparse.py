#!/usr/bin/env python3
"""Compare the original dense planner with the new sparse planner on the same map."""

import os
import sys
import pickle
import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root, 'planner', 'scripts'))
sys.path.append(os.path.join(root, 'planner'))

from planner_wrapper import TomogramPlanner
from sparse_planner_wrapper import SparseTomogramPlanner
from config import Config

# Same constants used during sparse export / search.
SPARSE_ASTAR_COST_THRESHOLD = 35.0
MIN_PAIR_DISTANCE = 5.0  # meters
NUM_PAIRS = 8


def path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))


def hausdorff_distance(a, b):
    """Bidirectional 3-D Hausdorff distance between two point sets."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return float('inf')
    d_ab = np.max(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2), axis=1))
    d_ba = np.max(np.min(np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2), axis=1))
    return max(d_ab, d_ba)


def dense_raw_to_world(dense_arrays, raw_grid):
    """Convert the dense A* result matrix [layer, row, col] to world coordinates."""
    if raw_grid is None or raw_grid.shape[0] == 0:
        return None
    resolution = dense_arrays['resolution']
    center = np.asarray(dense_arrays['center'], dtype=np.float64)
    elev_g = np.asarray(dense_arrays['data'][3], dtype=np.float32)
    slice_heights = np.asarray(dense_arrays['slice_heights'], dtype=np.float32)
    n_slice, map_dim_x, map_dim_y = elev_g.shape
    offset_x = map_dim_x / 2.0
    offset_y = map_dim_y / 2.0

    world = []
    for i, (layer, row, col) in enumerate(raw_grid):
        layer = int(layer)
        row = int(row)
        col = int(col)
        if layer < 0 or layer >= n_slice or row < 0 or row >= map_dim_y or col < 0 or col >= map_dim_x:
            continue
        z = elev_g[layer, row, col]
        if not np.isfinite(z):
            # Dense A* may pass through cells with no ground height; use the slice height as a fallback.
            z = slice_heights[layer]
        x = (col - offset_x) * resolution + center[0]
        y = (row - offset_y) * resolution + center[1]
        world.append([x, y, z])
    if len(world) == 0:
        return None
    return np.array(world, dtype=np.float64)


def _sparse_bfs_reachable(sparse_planner, start_idx, goal_idx, max_visit=200000):
    """Quick BFS on the sparse graph using the same rules as C++ SparseAstar."""
    idx_set = sparse_planner.sparse_index_map
    threshold = SPARSE_ASTAR_COST_THRESHOLD
    step_max = 0.5
    shape = (sparse_planner.n_slice, sparse_planner.map_dim[0], sparse_planner.map_dim[1])
    neigh2d = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neigh2d_c = [(0, 0)] + neigh2d

    start_key = tuple(int(v) for v in start_idx)
    goal_key = tuple(int(v) for v in goal_idx)
    if start_key not in idx_set or goal_key not in idx_set:
        return False

    from collections import deque
    q = deque([start_key])
    visited = {start_key}
    while q and len(visited) < max_visit:
        node = q.popleft()
        if node == goal_key:
            return True
        l, r, c = node
        i = idx_set[node]
        h = sparse_planner.elev_g[i]
        # Same layer
        for dr, dc in neigh2d:
            nb = (l, r + dr, c + dc)
            if nb in idx_set and nb not in visited:
                j = idx_set[nb]
                if sparse_planner.trav[j] <= threshold or sparse_planner.gateway[j] != 0:
                    if sparse_planner.trav[j] <= threshold or abs(sparse_planner.elev_g[j] - h) <= step_max:
                        visited.add(nb)
                        q.append(nb)
        # Cross layer
        g = sparse_planner.gateway[i]
        if g > 0 and l + 1 < shape[0]:
            for dr, dc in neigh2d_c:
                nb = (l + 1, r + dr, c + dc)
                if nb in idx_set and nb not in visited:
                    j = idx_set[nb]
                    if sparse_planner.trav[j] <= threshold or sparse_planner.gateway[j] != 0:
                        if abs(sparse_planner.elev_g[j] - h) <= step_max:
                            visited.add(nb)
                            q.append(nb)
                            break
        if g < 0 and l - 1 >= 0:
            for dr, dc in neigh2d_c:
                nb = (l - 1, r + dr, c + dc)
                if nb in idx_set and nb not in visited:
                    j = idx_set[nb]
                    if sparse_planner.trav[j] <= threshold or sparse_planner.gateway[j] != 0:
                        if abs(sparse_planner.elev_g[j] - h) <= step_max:
                            visited.add(nb)
                            q.append(nb)
                            break
    return False


def sample_start_goal(sparse_planner, rng, min_distance):
    """Sample a random start/goal pair from the same sparse connected component."""
    mask = (
        (sparse_planner.trav <= SPARSE_ASTAR_COST_THRESHOLD)
        & (sparse_planner.gateway == 0)
    )
    candidates = np.where(mask)[0]
    if len(candidates) < 2:
        return None

    offset_x = sparse_planner.map_dim[1] / 2.0
    offset_y = sparse_planner.map_dim[0] / 2.0

    def idx_to_world(idx):
        layer, row, col = sparse_planner.indices[idx]
        x = (col - offset_x) * sparse_planner.resolution + sparse_planner.center[0]
        y = (row - offset_y) * sparse_planner.resolution + sparse_planner.center[1]
        z = sparse_planner.elev_g[idx]
        return np.array([x, y, z], dtype=np.float64), np.array([layer, row, col], dtype=np.int32)

    for _ in range(2000):
        i, j = rng.choice(candidates, size=2, replace=False)
        start_world, start_idx = idx_to_world(i)
        goal_world, goal_idx = idx_to_world(j)
        if np.linalg.norm(start_world[:2] - goal_world[:2]) < min_distance:
            continue
        if _sparse_bfs_reachable(sparse_planner, start_idx, goal_idx):
            return start_world, goal_world, start_idx, goal_idx
    return None


def main():
    cfg = Config()

    dense_path = os.path.join(root, 'rsc', 'tomogram', 'scene_map.pickle')
    sparse_path = os.path.join(root, 'rsc', 'tomogram', 'scene_map_sparse.pickle')

    print("Loading dense pickle arrays...")
    with open(dense_path, 'rb') as f:
        dense_arrays = pickle.load(f)

    print("Loading dense planner...")
    dense_planner = TomogramPlanner(cfg)
    dense_planner.loadTomogram('scene_map')

    print("Loading sparse planner...")
    sparse_planner = SparseTomogramPlanner(cfg)
    sparse_planner.loadTomogram('scene_map_sparse')

    rng = np.random.default_rng(seed=42)

    results = []
    pair_id = 0
    while pair_id < NUM_PAIRS:
        sample = sample_start_goal(sparse_planner, rng, MIN_PAIR_DISTANCE)
        if sample is None:
            print("Not enough sparse nodes to sample pairs.")
            break
        start_world, goal_world, start_idx, goal_idx = sample

        print(f"\n=== Pair {pair_id + 1}/{NUM_PAIRS} ===")
        print(f"start_world = {start_world.tolist()}")
        print(f"goal_world  = {goal_world.tolist()}")
        print(f"start_idx   = {start_idx.tolist()}")
        print(f"goal_idx    = {goal_idx.tolist()}")

        # Dense planner (original pipeline: raw A* + GPMP optimization).
        try:
            dense_traj = dense_planner.plan(
                start_world[:2], goal_world[:2],
                start_world[2] + 0.5, goal_world[2] + 0.5
            )
        except Exception as e:
            print(f"Dense planner exception: {e}")
            dense_traj = None

        try:
            dense_raw_grid = dense_planner.planner.get_debug_path()
            dense_raw = dense_raw_to_world(dense_arrays, dense_raw_grid)
        except Exception as e:
            print(f"Dense raw path extraction exception: {e}")
            dense_raw = None

        # Sparse planner (raw Sparse A*) using the known sparse indices.
        try:
            sparse_path_result = sparse_planner.plan_from_indices(start_idx, goal_idx)
        except Exception as e:
            print(f"Sparse planner exception: {e}")
            sparse_path_result = None

        dense_success = dense_traj is not None and len(dense_traj) > 0
        dense_raw_success = dense_raw is not None and len(dense_raw) > 0
        sparse_success = sparse_path_result is not None and len(sparse_path_result) > 0

        result = {
            'pair_id': pair_id,
            'start': start_world,
            'goal': goal_world,
            'dense_success': dense_success,
            'dense_raw_success': dense_raw_success,
            'sparse_success': sparse_success,
            'dense_len': path_length(dense_traj) if dense_success else float('inf'),
            'dense_raw_len': path_length(dense_raw) if dense_raw_success else float('inf'),
            'sparse_len': path_length(sparse_path_result) if sparse_success else float('inf'),
            'dense_nodes': len(dense_traj) if dense_success else 0,
            'dense_raw_nodes': len(dense_raw) if dense_raw_success else 0,
            'sparse_nodes': len(sparse_path_result) if sparse_success else 0,
            'dense_endpoint_err': (np.linalg.norm(dense_traj[-1] - goal_world) if dense_success else float('inf')),
            'dense_raw_endpoint_err': (np.linalg.norm(dense_raw[-1] - goal_world) if dense_raw_success else float('inf')),
            'sparse_endpoint_err': (np.linalg.norm(sparse_path_result[-1] - goal_world) if sparse_success else float('inf')),
            'hausdorff_dense_raw_vs_sparse': (
                hausdorff_distance(dense_raw, sparse_path_result)
                if dense_raw_success and sparse_success else float('inf')
            ),
        }
        results.append(result)

        print(f"dense success = {dense_success}, len = {result['dense_len']:.2f}, nodes = {result['dense_nodes']}")
        print(f"dense_raw success = {dense_raw_success}, len = {result['dense_raw_len']:.2f}, nodes = {result['dense_raw_nodes']}")
        print(f"sparse success = {sparse_success}, len = {result['sparse_len']:.2f}, nodes = {result['sparse_nodes']}")
        if sparse_success and dense_raw_success:
            print(f"hausdorff(raw_dense, sparse) = {result['hausdorff_dense_raw_vs_sparse']:.3f}")

        out_dir = os.path.join(root, 'tests', 'results')
        os.makedirs(out_dir, exist_ok=True)
        if dense_success:
            np.savetxt(os.path.join(out_dir, f'pair_{pair_id:02d}_dense_gpmp.csv'), dense_traj, delimiter=',', header='x,y,z')
        if dense_raw_success:
            np.savetxt(os.path.join(out_dir, f'pair_{pair_id:02d}_dense_raw.csv'), dense_raw, delimiter=',', header='x,y,z')
        if sparse_success:
            np.savetxt(os.path.join(out_dir, f'pair_{pair_id:02d}_sparse.csv'), sparse_path_result, delimiter=',', header='x,y,z')

        pair_id += 1

    # Summary table.
    print("\n=== Summary ===")
    print(f"{'pair':>4} {'dense':>5} {'raw':>5} {'sparse':>6} {'d_len':>8} {'dr_len':>8} {'s_len':>8} {'d_end':>8} {'dr_end':>8} {'s_end':>8} {'hausdorff':>10}")
    for r in results:
        print(
            f"{r['pair_id']:>4} "
            f"{int(r['dense_success']):>5} "
            f"{int(r['dense_raw_success']):>5} "
            f"{int(r['sparse_success']):>6} "
            f"{r['dense_len']:>8.2f} "
            f"{r['dense_raw_len']:>8.2f} "
            f"{r['sparse_len']:>8.2f} "
            f"{r['dense_endpoint_err']:>8.2f} "
            f"{r['dense_raw_endpoint_err']:>8.2f} "
            f"{r['sparse_endpoint_err']:>8.2f} "
            f"{r['hausdorff_dense_raw_vs_sparse']:>10.3f}"
        )

    summary_path = os.path.join(root, 'tests', 'results', 'summary.pickle')
    with open(summary_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
