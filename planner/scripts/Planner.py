#!/usr/bin/env python3
"""Simple standalone path planner on a sparse tomogram pickle.

Wraps SparseTomogramPlanner and exposes a small CLI:
    python3 Planner.py --start -100 -10 0 --goal -90 0 0

The default map is the filtered sparse pickle produced by
pcd_to_filtered_sparse.py: `scene_map_sparse_planner.pickle`.
"""

import argparse
import os
import pickle
import sys

import numpy as np

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

    def _snap_to_node(self, pos):
        """Return the grid index (layer, row, col) of the nearest sparse node."""
        pos = np.asarray(pos, dtype=np.float64)
        cols = self.planner.indices[:, 2].astype(np.float64)
        rows = self.planner.indices[:, 1].astype(np.float64)
        zs = self.planner.elev_g
        xs = (cols - self.planner.offset[0]) * self.planner.resolution + self.planner.center[0]
        ys = (rows - self.planner.offset[1]) * self.planner.resolution + self.planner.center[1]
        diffs = np.stack([xs, ys, zs], axis=1) - pos
        dists = np.linalg.norm(diffs, axis=1)
        i = int(np.argmin(dists))
        return self.planner.indices[i].copy(), float(dists[i])

    def plan(self, start, goal):
        """Plan a path.

        Args:
            start: [x, y, z] world coordinates.
            goal:  [x, y, z] world coordinates.

        Returns:
            Nx3 numpy array of path waypoints, or None if planning fails.
        """
        start_idx, start_dist = self._snap_to_node(start)
        goal_idx, goal_dist = self._snap_to_node(goal)
        print(f"snapped_start_idx = {start_idx.tolist()}, distance = {start_dist:.3f}")
        print(f"snapped_goal_idx  = {goal_idx.tolist()}, distance = {goal_dist:.3f}")
        return self.planner.plan_from_indices(start_idx, goal_idx)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan a 3-D path on a sparse tomogram pickle."
    )
    parser.add_argument(
        '--map', type=str, default='scene_map_sparse_planner',
        help="sparse tomogram pickle name under rsc/tomogram/ (without .pickle)"
    )
    parser.add_argument(
        '--start', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'),
        help="start position in world coordinates"
    )
    parser.add_argument(
        '--goal', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'),
        help="goal position in world coordinates"
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help="optional output file (.npy, .pickle, .pcd, .csv, or text)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading map: {args.map}.pickle")
    planner = Planner(map_name=args.map)

    print(f"Planning from {args.start} to {args.goal} ...")
    path = planner.plan(args.start, args.goal)

    if path is None:
        print("Planning failed.")
        return 1

    print(f"Path waypoints: {path.shape[0]}")
    print(f"Path length: {path_length(path):.3f} m")

    if args.output:
        save_path(path, args.output)
        print(f"Saved path to: {args.output}")
    else:
        print("Path (x y z):")
        for p in path:
            print(f"  {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
