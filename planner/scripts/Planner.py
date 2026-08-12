#!/usr/bin/env python3
"""Simple standalone path planner on a sparse tomogram pickle.

Wraps SparseTomogramPlanner and exposes a small CLI:
    python3 Planner.py --start -100 -10 0 --goal -90 0 0
    python3 Planner.py --json waypoints.json --start-node desk --goal-node outdoor
    python3 Planner.py --json pairs.json --pair-index 0 --output path.json
    python3 Planner.py --json pairs.json --all-pairs --output all_paths.json
    python3 Planner.py --json pairs.json --all-pairs --jobs 8 --output all_paths.json

With --all-pairs, planning is batched into a single C++ call
(SparseAstar::SearchBatch): pairs are searched concurrently inside the C++
layer, each thread keeping its own A* scratch while sharing the map graph
read-only. --jobs sets the number of C++ worker threads (default: hardware
concurrency). Note: batch paths may differ slightly from one-at-a-time
planning because A* open-set tie-breaking differs; both are valid and the
batch result is deterministic across thread counts.

With --json, the file can be either:
  - a waypoint list: [{"name": ..., "pose": [x, y, z], ...}, ...]
  - a task pair list: [{"start": [...], "goal": [...], ...}, ...]

When the input is JSON, the output defaults to JSON format with fields:
  {"start_pose": [...], "goal_pose": [...], "trajectory": [[...], ...]}.
With --all-pairs, all results are saved as a single JSON array.
Use --format csv to force CSV output for single-pair planning.

The default map is the filtered sparse pickle produced by
pcd_to_filtered_sparse.py: `scene_map_sparse_planner.pickle`.
The result is saved to `path.csv` next to this script unless --output
is given.
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

# Make planner/ visible so `config` and `sparse_planner_wrapper` can be imported.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

# Ensure the C++ shared libraries can be found before loading the pybind module.
RSG_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
LIB_DIR = os.path.join(RSG_ROOT, 'planner', 'lib')
GTSAM_DIR = os.path.join(LIB_DIR, '3rdparty', 'gtsam-4.1.1', 'install', 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{LIB_DIR}:{GTSAM_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"

from config import Config
from sparse_planner_wrapper import SparseTomogramPlanner


class Planner(object):
    """Thin wrapper around SparseTomogramPlanner for offline path queries."""

    def __init__(self, map_name='scene_map_sparse_planner'):
        self.cfg = Config()
        self.planner = SparseTomogramPlanner(self.cfg)
        self.planner.loadTomogram(map_name)

        # Precompute world coordinates of all sparse nodes once.
        cols = self.planner.indices[:, 2].astype(np.float64)
        rows = self.planner.indices[:, 1].astype(np.float64)
        xs = (cols - self.planner.offset[0]) * self.planner.resolution + self.planner.center[0]
        ys = (rows - self.planner.offset[1]) * self.planner.resolution + self.planner.center[1]
        self.node_xyz = np.stack([xs, ys, self.planner.elev_g], axis=1)
        self._node_tree = cKDTree(self.node_xyz)
        self._snap_cache: dict[tuple[float, float, float], tuple[np.ndarray, float]] = {}

    def _snap_to_node(self, pos):
        """Return the grid index (layer, row, col) of the nearest sparse node."""
        pos = np.asarray(pos, dtype=np.float64)
        key = (float(pos[0]), float(pos[1]), float(pos[2]))
        if key in self._snap_cache:
            return self._snap_cache[key]
        dist, i = self._node_tree.query(pos, k=1)
        result = (self.planner.indices[i].copy(), float(dist))
        self._snap_cache[key] = result
        return result

    def plan(self, start, goal):
        """Plan a path.

        Args:
            start: [x, y, z] world coordinates.
            goal:  [x, y, z] world coordinates.

        Returns:
            Nx3 numpy array of path waypoints, or None if planning fails.
        """
        t0 = time.perf_counter()
        start_idx, start_dist = self._snap_to_node(start)
        t1 = time.perf_counter()
        goal_idx, goal_dist = self._snap_to_node(goal)
        t2 = time.perf_counter()
        print(f"snapped_start_idx = {start_idx.tolist()}, distance = {start_dist:.3f} ({(t1 - t0)*1000:.1f} ms)")
        print(f"snapped_goal_idx  = {goal_idx.tolist()}, distance = {goal_dist:.3f} ({(t2 - t1)*1000:.1f} ms)")
        return self.planner.plan_from_indices(start_idx, goal_idx)

    def plan_batch(self, pairs, num_threads=0):
        """Plan many (start, goal) world pairs in one C++ call.

        Snapping mirrors plan(): nearest sparse node, then a traversability
        ring snap. The A* searches run concurrently inside the C++ layer
        (SparseAstar::SearchBatch); the map graph is shared read-only and each
        thread keeps its own scratch, so no processes or IPC are involved.

        Args:
            pairs: list of dicts with 'start' and 'goal' ([x, y, z] world).
            num_threads: C++ worker threads; <= 0 uses hardware_concurrency().

        Returns:
            A list aligned with pairs; each entry is an Nx3 path array or None
            when planning fails for that pair.
        """
        results = [None] * len(pairs)
        valid_starts = []
        valid_goals = []
        valid_idx = []
        for i, pair in enumerate(pairs):
            start_idx, _ = self._snap_to_node(pair['start'])
            goal_idx, _ = self._snap_to_node(pair['goal'])
            s, _ = self.planner._snap(start_idx)
            g, _ = self.planner._snap(goal_idx)
            if s is None or g is None:
                continue
            valid_starts.append(s)
            valid_goals.append(g)
            valid_idx.append(i)
        if valid_idx:
            paths = self.planner.plan_batch_from_indices(
                np.array(valid_starts, dtype=np.int32),
                np.array(valid_goals, dtype=np.int32),
                num_threads,
            )
            for i, path in zip(valid_idx, paths):
                results[i] = path
        return results


def load_waypoint_pose(json_path, node_name):
    """Return the [x, y, z] pose of a named waypoint from a JSON waypoint file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    for entry in data:
        if entry.get('name') == node_name:
            pose = entry.get('pose')
            if isinstance(pose, list) and len(pose) == 3:
                return pose
            raise ValueError(f"waypoint {node_name!r} has no valid 'pose' [x, y, z]")
    available = [entry.get('name') for entry in data]
    raise KeyError(
        f"waypoint {node_name!r} not found in {json_path}; available: {available}"
    )


def _is_pair_json(data):
    """Return True if the loaded JSON follows the task pair format."""
    return (
        isinstance(data, list)
        and len(data) > 0
        and isinstance(data[0], dict)
        and 'start' in data[0]
        and 'goal' in data[0]
    )


def load_start_goal_from_json(json_path, pair_index=0):
    """Return (start, goal) from a task pair JSON file.

    The file is expected to be a list of dicts with 'start' and 'goal' keys,
    each containing [x, y, z].
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not _is_pair_json(data):
        raise ValueError(
            f"{json_path} does not look like a pair JSON; "
            "expected a list of {{'start': [x,y,z], 'goal': [x,y,z]}} objects"
        )

    if not (0 <= pair_index < len(data)):
        raise IndexError(
            f"pair_index {pair_index} out of range for {len(data)} pairs in {json_path}"
        )

    pair = data[pair_index]
    start = pair.get('start')
    goal = pair.get('goal')
    if not (isinstance(start, list) and len(start) == 3):
        raise ValueError(f"pair {pair_index} has invalid 'start' {start}")
    if not (isinstance(goal, list) and len(goal) == 3):
        raise ValueError(f"pair {pair_index} has invalid 'goal' {goal}")
    return start, goal


def parse_xyz(arg):
    """Parse an argument list of three floats."""
    parts = [float(x) for x in arg]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected exactly 3 floats: x y z")
    return parts


def path_length(path):
    if path is None or path.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def save_path(path, output_path):
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.npy':
        np.save(output_path, path)
    elif ext == '.pickle' or ext == '.pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(path, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif ext == '.pcd':
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(path)
        o3d.io.write_point_cloud(output_path, pcd)
    elif ext == '.csv':
        np.savetxt(output_path, path, delimiter=',', header='x,y,z', comments='')
    else:
        # Default to numpy text format.
        np.savetxt(output_path, path, delimiter=' ', header='x y z', comments='')


def _to_float_list(arr):
    """Convert a numpy array or list to a plain Python list of floats."""
    return [float(x) for x in arr]


def build_trajectory_record(
    start,
    goal,
    trajectory,
    start_name=None,
    goal_name=None,
    pair_index=None,
    extra=None,
):
    """Build a trajectory record matching the task_point_traj JSON format."""
    record = {
        "start_pose": _to_float_list(start),
        "goal_pose": _to_float_list(goal),
        "trajectory": [_to_float_list(p) for p in trajectory],
    }
    if start_name is not None:
        record["start_name"] = start_name
    if goal_name is not None:
        record["goal_name"] = goal_name
    if pair_index is not None:
        record["pair_index"] = pair_index
    if extra is not None:
        record.update(extra)
    return record


def save_trajectory_json(record, output_path):
    """Save a single trajectory record or a list of records to JSON."""
    with open(output_path, 'w') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def _output_is_json(output_path):
    return os.path.splitext(output_path)[1].lower() == '.json'


def _use_json(fmt, output_path, explicit_format, is_all_pairs=False):
    """Return True if the output should be JSON.

    Args:
        fmt: 'pair', 'waypoint', or 'raw'.
        output_path: the --output value.
        explicit_format: the --format value (auto, csv, or json).
        is_all_pairs: whether --all-pairs is enabled (output is a directory or
            a combined file).
    """
    if explicit_format == 'json':
        return True
    if explicit_format == 'csv':
        return False
    # auto
    if _output_is_json(output_path):
        return True
    if fmt == 'pair' and is_all_pairs:
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan a 3-D path on a sparse tomogram pickle."
    )
    parser.add_argument(
        '--map', type=str, default='scene_map_sparse_planner',
        help="sparse tomogram pickle name under rsc/tomogram/ (without .pickle)"
    )
    parser.add_argument(
        '--start', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'),
        help="start position in world coordinates"
    )
    parser.add_argument(
        '--goal', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'),
        help="goal position in world coordinates"
    )
    parser.add_argument(
        '--json', type=str, default=None,
        help="JSON file: either a waypoint list with 'name'/'pose' fields, "
             "or a task pair list with 'start'/'goal' fields"
    )
    parser.add_argument(
        '--start-node', type=str, default=None,
        help="name of the start waypoint in --json (waypoint JSON only)"
    )
    parser.add_argument(
        '--goal-node', type=str, default=None,
        help="name of the goal waypoint in --json (waypoint JSON only)"
    )
    parser.add_argument(
        '--pair-index', type=int, default=0,
        help="index of the pair to use from a task pair JSON (default: 0)"
    )
    parser.add_argument(
        '--all-pairs', action='store_true',
        help="plan all pairs from a task pair JSON; results are saved as a single JSON array"
    )
    parser.add_argument(
        '--jobs', type=int, default=None,
        help="number of C++ worker threads for --all-pairs (default: hardware concurrency)"
    )
    parser.add_argument(
        '--output', type=str, default=os.path.join(SCRIPT_DIR, 'path.csv'),
        help="output file (.json, .npy, .pickle, .pcd, .csv, or text). "
             "When --all-pairs is used, --output is a single JSON file. Defaults to path.csv"
    )
    parser.add_argument(
        '--format', type=str, default='auto', choices=['auto', 'csv', 'json'],
        help="output format: auto (infer from --output extension or input type), "
             "csv, or json"
    )
    return parser.parse_args()


def _detect_json_format(json_path):
    """Detect whether the JSON file is waypoint format or pair format."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if _is_pair_json(data):
        return 'pair'
    if isinstance(data, list) and len(data) > 0 and 'name' in data[0] and 'pose' in data[0]:
        return 'waypoint'
    return 'unknown'


def main():
    args = parse_args()

    if args.json:
        fmt = _detect_json_format(args.json)
        if fmt == 'pair':
            if args.start_node or args.goal_node:
                print("ERROR: --start-node/--goal-node are not used with pair JSON; "
                      "use --pair-index or --all-pairs instead")
                return 1
        elif fmt == 'waypoint':
            if not (args.start_node and args.goal_node):
                print("ERROR: waypoint JSON requires --start-node and --goal-node")
                return 1
        else:
            print("ERROR: --json format not recognized; expected waypoint list or pair list")
            return 1
    elif args.start is not None and args.goal is not None:
        fmt = 'raw'
    else:
        print("ERROR: provide either --start/--goal XYZ or --json")
        return 1

    if fmt == 'pair' and args.all_pairs:
        with open(args.json, 'r') as f:
            pairs = json.load(f)

        print(f"Loading map: {args.map}.pickle")
        planner = Planner(map_name=args.map)

        jobs = args.jobs or 0
        print(f"Planning {len(pairs)} pairs with {jobs or 'auto'} C++ worker thread(s) ...",
              flush=True)
        t0 = time.perf_counter()
        try:
            paths = planner.plan_batch(pairs, jobs)
        except KeyboardInterrupt:
            print(f"Interrupted: batch canceled after {time.perf_counter() - t0:.1f} s "
                  f"(no output written to {args.output})")
            return 130
        elapsed = time.perf_counter() - t0

        # Per-pair progress is printed live by the C++ layer as each pair
        # finishes; here we only aggregate the results for the output file.
        records = []
        success_count = 0
        for idx, path in enumerate(paths):
            pair = pairs[idx]
            if path is None:
                continue
            extra = {k: v for k, v in pair.items() if k not in ('start', 'goal')}
            records.append(build_trajectory_record(
                pair['start'], pair['goal'], path, pair_index=idx, extra=extra
            ))
            success_count += 1

        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        save_trajectory_json(records, args.output)
        print(f"Finished: {success_count}/{len(pairs)} pairs planned in {elapsed:.2f} s -> {args.output}")
        return 0

    print(f"Loading map: {args.map}.pickle")
    planner = Planner(map_name=args.map)

    if fmt == 'pair':
        with open(args.json, 'r') as f:
            pairs = json.load(f)
        pair = pairs[args.pair_index]
        start = pair.get('start')
        goal = pair.get('goal')
        extra = {k: v for k, v in pair.items() if k not in ('start', 'goal')}
    elif fmt == 'waypoint':
        start = load_waypoint_pose(args.json, args.start_node)
        goal = load_waypoint_pose(args.json, args.goal_node)
        extra = None
    else:
        start, goal = args.start, args.goal
        extra = None

    print(f"Planning from {start} to {goal} ...")
    path = planner.plan(start, goal)

    if path is None:
        print("Planning failed.")
        return 1

    print(f"Path waypoints: {path.shape[0]}")
    print(f"Path length: {path_length(path):.3f} m")

    if args.output:
        use_json = _use_json(fmt, args.output, args.format, is_all_pairs=False)
        if use_json:
            start_name = args.start_node if fmt == 'waypoint' else None
            goal_name = args.goal_node if fmt == 'waypoint' else None
            pair_index = args.pair_index if fmt == 'pair' else None
            record = build_trajectory_record(
                start, goal, path,
                start_name=start_name,
                goal_name=goal_name,
                pair_index=pair_index,
                extra=extra,
            )
            save_trajectory_json(record, args.output)
        else:
            save_path(path, args.output)
        print(f"Saved path to: {args.output}")
    else:
        print("Path (x y z):")
        for p in path:
            print(f"  {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
