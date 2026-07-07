# PctPlanner — Project Notes

## Overview

ROS 2 Humble planning stack that builds a GPU-accelerated voxel tomogram from a
PCD file and plans 3-D paths with Sparse A* on a sparse tomogram graph.

This branch is **sparse-only**: the planner no longer uses the dense grid map,
dense A*, `DenseElevationMap`, `OfflineElePlanner.init_map`, GPMP, or
`GenerateTrajectory`.

## Key directories and files

| Path | Purpose |
|------|---------|
| `tomography/scripts/tomography.py` | PCD → sparse tomogram export (`scene_map_sparse.pickle`) |
| `planner/lib/src/sparse_a_star/` | C++ `SparseAstar` implementation and pybind interface |
| `planner/scripts/sparse_planner_wrapper.py` | Python `SparseTomogramPlanner` |
| `planner/scripts/plan.py` | ROS node: loads sparse map, waits for `/start_pos`/`/end_pos`, publishes `/pct_path` |
| `planner/scripts/plan_direct.py` | ROS node: loads sparse map directly, publishes `/pct_path2` |
| `planner/scripts/plan_systemt.py` | ROS node: system pose-driven replanning on `/local_pose`, publishes `/pct_path2` |
| `rsc/tomogram/scene_map_sparse.pickle` | Output sparse map |
| `planner/build.sh` | Build C++ pybind modules |

| `rsc/tomogram/publish_sparse_tomogram.py` | Optional PointCloud2 publisher for the sparse nodes |

## Build

```bash
cd /home/nuc/numa/PctPlanner/planner
./build_thirdparty.sh   # only once
./build.sh
```

After building, export library paths for the current shell:

```bash
export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/nuc/numa/PctPlanner/planner/lib:$PYTHONPATH
```

## Run

Generate the sparse map:

```bash
cd /home/nuc/numa/PctPlanner/tomography/scripts
python3 tomography.py
```

Plan from published start/end points:

```bash
cd /home/nuc/numa/PctPlanner/planner/scripts
python3 plan_direct.py
# In another terminal:
# cd /home/nuc/numa/PctPlanner/rsc && python3 publish_start_end_pos.py
```

Visualize the sparse planning nodes:

```bash
cd /home/nuc/numa/PctPlanner/rsc/tomogram
python3 publish_sparse_tomogram.py
```

System pose-driven planning:

```bash
python3 plan_systemt.py
```

## Sparse map contract

The file `rsc/tomogram/scene_map_sparse.pickle` must contain:

```python
{
    "format": "tomogram_sparse_v1",
    "shape": [n_slice, map_dim_y, map_dim_x],
    "resolution": float,
    "center": [cx, cy],
    "slice_h0": float,
    "slice_dh": float,
    "slice_heights": np.ndarray,
    "indices": np.ndarray[N, 3],   # int32 [layer, row, col]
    "trav": np.ndarray[N],         # float32
    "elev_g": np.ndarray[N],       # float32
    "elev_c": np.ndarray[N],       # float32
    "gateway": np.ndarray[N],      # int32, 0 / 2 / -2
}
```

There is **no** `"data"` field. `indices` is the single source of planning nodes.

## Invariants

- Thresholds in `tomography/scripts/tomography.py` must match
  `planner/scripts/sparse_planner_wrapper.py`:
  - `SPARSE_ASTAR_COST_THRESHOLD = 35.0`
  - `SPARSE_ROBOT_HEIGHT_MIN = 0.6`
  - `SPARSE_GATEWAY_COST_DELTA = 8.0`
  - `SPARSE_GATEWAY_HEIGHT_DELTA = 0.1`
- The sparse export must not perform secondary dilation or downsampling.
- Core ROS planning scripts (`plan.py`, `plan_direct.py`, `plan_systemt.py`) load
  only `scene_map_sparse.pickle` and use `SparseTomogramPlanner`.
- No `OfflineElePlanner.init_map`, `DenseElevationMap`, `GPMPOptimizer`, or
  `GenerateTrajectory` calls in the core ROS flow.
- Paths returned are raw Sparse A* polylines; no trajectory optimization is
  performed.

## Verification checklist

- [ ] `python3 tomography.py` produces `rsc/tomogram/scene_map_sparse.pickle`
      and **not** `scene_map.pickle`.
- [ ] Pickle has `"format": "tomogram_sparse_v1"` and no `"data"` key.
- [ ] Console logs show `dense_cells`, `sparse_nodes`, `sparse_ratio < 0.6`.
- [ ] `python3 plan_direct.py` prints the required snap logs:
      `raw_start_idx`, `snapped_start_idx`, `start_snap_distance`,
      `raw_goal_idx`, `snapped_goal_idx`, `goal_snap_distance`.
- [ ] It also prints `sparse_astar_init_ms`, `sparse_astar_search_ms`,
      `visited_nodes`, `path_nodes`.
- [ ] RViz path starts/ends at requested positions, uses `elev_g` z values, and
      is an unoptimized polyline.
