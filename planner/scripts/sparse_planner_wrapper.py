import os
import sys
import pickle
import numpy as np

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'
sys.path.append(rsg_root + '/planner/lib')

import sparse_a_star

# These must stay in sync with tomography/scripts/tomography.py.
SPARSE_ASTAR_COST_THRESHOLD = 35.0
SPARSE_ROBOT_HEIGHT_MIN = 0.6
SPARSE_GATEWAY_COST_DELTA = 8.0
SPARSE_GATEWAY_HEIGHT_DELTA = 0.1

SPARSE_MAX_SNAP_RADIUS = 80      # grid cells
SPARSE_STEP_MAX = 0.5            # meters
SPARSE_COST_WEIGHT = 0.2

TRAJECTORY_Z_OFFSET = 0.3        # meters, z lift applied to the planned trajectory


class SparseTomogramPlanner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tomo_dir = rsg_root + cfg.wrapper.tomo_dir

        self.resolution = None
        self.center = None
        self.n_slice = None
        self.slice_heights = None
        self.map_dim = None          # [map_dim_y, map_dim_x]
        self.offset = None           # [offset_x, offset_y]

        self.indices = None
        self.trav = None
        self.elev_g = None
        self.elev_c = None
        self.gateway = None
        self.sparse_index_map = {}

        self.astar = sparse_a_star.SparseAstar(sparse_a_star.SparseHeuristicType.DIAGONAL)

    def loadTomogram(self, tomo_file):
        file_path = self.tomo_dir + tomo_file + '.pickle'
        with open(file_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        self._initialize_from_dict(data_dict)

    def _initialize_from_dict(self, data_dict):
        if data_dict.get('format') != 'tomogram_sparse_v1':
            raise ValueError('expected tomogram_sparse_v1')

        self.resolution = float(data_dict['resolution'])
        self.center = np.asarray(data_dict['center'], dtype=np.float64)
        self.n_slice = int(data_dict['shape'][0])
        self.map_dim = [int(data_dict['shape'][1]), int(data_dict['shape'][2])]
        self.offset = np.array([self.map_dim[1] / 2.0, self.map_dim[0] / 2.0], dtype=np.float64)
        self.slice_heights = np.asarray(data_dict['slice_heights'], dtype=np.float32)

        self.indices = np.asarray(data_dict['indices'], dtype=np.int32)
        self.trav = np.asarray(data_dict['trav'], dtype=np.float64)
        self.elev_g = np.asarray(data_dict['elev_g'], dtype=np.float64)
        self.elev_c = np.asarray(data_dict['elev_c'], dtype=np.float64)
        self.gateway = np.asarray(data_dict['gateway'], dtype=np.int32)

        assert self.indices.ndim == 2 and self.indices.shape[1] == 3
        assert (
            len(self.trav) == len(self.elev_g) == len(self.elev_c) == len(self.gateway)
            == self.indices.shape[0]
        )

        self.sparse_index_map = {
            (int(self.indices[i, 0]), int(self.indices[i, 1]), int(self.indices[i, 2])): i
            for i in range(self.indices.shape[0])
        }

        shape_array = np.array([self.n_slice, self.map_dim[0], self.map_dim[1]], dtype=np.int32)
        self.astar.init(
            shape_array,
            self.resolution,
            SPARSE_ASTAR_COST_THRESHOLD,
            SPARSE_STEP_MAX,
            SPARSE_COST_WEIGHT,
            self.indices,
            self.trav,
            self.elev_g,
            self.elev_c,
            self.gateway,
        )

    def z2slice(self, z):
        slice_idx = np.searchsorted(self.slice_heights, z, side='right') - 1
        return int(np.clip(slice_idx, 0, self.n_slice - 1))

    def _world_to_grid(self, xy, z):
        x, y = xy[0], xy[1]
        col = int(np.round((x - self.center[0]) / self.resolution + self.offset[0]))
        row = int(np.round((y - self.center[1]) / self.resolution + self.offset[1]))
        layer = self.z2slice(z)
        return np.array([layer, row, col], dtype=np.int32)

    def _snap(self, raw_idx):
        raw_layer, raw_row, raw_col = raw_idx
        best_idx = None
        best_dist_sq = float('inf')
        best_cost = float('inf')

        for radius in range(SPARSE_MAX_SNAP_RADIUS + 1):
            # Iterate over the square ring of the given Chebyshev radius.
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    row = raw_row + dr
                    col = raw_col + dc
                    key = (int(raw_layer), int(row), int(col))
                    idx = self.sparse_index_map.get(key)
                    if idx is None:
                        continue
                    if self.trav[idx] > SPARSE_ASTAR_COST_THRESHOLD and self.gateway[idx] == 0:
                        continue
                    dist_sq = dr * dr + dc * dc
                    cost = self.trav[idx]
                    if dist_sq < best_dist_sq or (dist_sq == best_dist_sq and cost < best_cost):
                        best_dist_sq = dist_sq
                        best_cost = cost
                        best_idx = np.array([raw_layer, row, col], dtype=np.int32)
            if best_idx is not None:
                break

        if best_idx is None:
            return None, float('inf')
        return best_idx, np.sqrt(best_dist_sq)

    def plan_from_indices(self, start_idx, goal_idx):
        """Plan directly from raw grid indices (used by comparison tests)."""
        raw_start = np.asarray(start_idx, dtype=np.int32)
        raw_goal = np.asarray(goal_idx, dtype=np.int32)

        snapped_start, start_snap_distance = self._snap(raw_start)
        snapped_goal, goal_snap_distance = self._snap(raw_goal)

        print(f"raw_start_idx = {raw_start.tolist()}")
        print(f"snapped_start_idx = {None if snapped_start is None else snapped_start.tolist()}")
        print(f"start_snap_distance = {start_snap_distance:.2f}")
        print(f"raw_goal_idx = {raw_goal.tolist()}")
        print(f"snapped_goal_idx = {None if snapped_goal is None else snapped_goal.tolist()}")
        print(f"goal_snap_distance = {goal_snap_distance:.2f}")

        if snapped_start is None or snapped_goal is None:
            print("SparseTomogramPlanner: start or goal snap failed")
            return None

        return self._search_and_convert(snapped_start, snapped_goal)

    def plan(self, start_pos, end_pos, start_z=0.0, end_z=0.0):
        start_pos = np.asarray(start_pos, dtype=np.float64)
        end_pos = np.asarray(end_pos, dtype=np.float64)

        raw_start = self._world_to_grid(start_pos, start_z)
        raw_goal = self._world_to_grid(end_pos, end_z)

        snapped_start, start_snap_distance = self._snap(raw_start)
        snapped_goal, goal_snap_distance = self._snap(raw_goal)

        print(f"raw_start_idx = {raw_start.tolist()}")
        print(f"snapped_start_idx = {None if snapped_start is None else snapped_start.tolist()}")
        print(f"start_snap_distance = {start_snap_distance:.2f}")
        print(f"raw_goal_idx = {raw_goal.tolist()}")
        print(f"snapped_goal_idx = {None if snapped_goal is None else snapped_goal.tolist()}")
        print(f"goal_snap_distance = {goal_snap_distance:.2f}")

        if snapped_start is None or snapped_goal is None:
            print("SparseTomogramPlanner: start or goal snap failed")
            return None

        return self._search_and_convert(snapped_start, snapped_goal)

    def _search_and_convert(self, snapped_start, snapped_goal):
        success = self.astar.search(snapped_start, snapped_goal)
        if not success:
            print("SparseTomogramPlanner: search failed")
            return None

        path_grid = self.astar.get_result_matrix()
        if path_grid.shape[0] == 0:
            return None

        path_xyz = np.zeros((path_grid.shape[0], 3), dtype=np.float64)
        for i, (layer, row, col) in enumerate(path_grid):
            idx = self.sparse_index_map[(int(layer), int(row), int(col))]
            x = (col - self.offset[0]) * self.resolution + self.center[0]
            y = (row - self.offset[1]) * self.resolution + self.center[1]
            z = self.elev_g[idx] + TRAJECTORY_Z_OFFSET
            path_xyz[i] = [x, y, z]

        return path_xyz
