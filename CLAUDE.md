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
| `planner/lib/src/sparse_a_star/` | C++ `SparseAstar` implementation, pybind interface, and batch `SearchBatch` |
| `planner/scripts/sparse_planner_wrapper.py` | Python `SparseTomogramPlanner` (`plan_batch_from_indices` for batched planning) |
| `planner/scripts/Planner.py` | Offline CLI path planner on a sparse pickle (`--jobs` for batched planning) |
| `planner/scripts/plan.py` | ROS node: loads sparse map, waits for `/start_pos`/`/end_pos`, publishes `/pct_path` |
| `planner/scripts/plan_direct.py` | ROS node: loads sparse map directly, publishes `/pct_path2` |
| `planner/scripts/plan_systemt.py` | ROS node: system pose-driven replanning on `/local_pose`, publishes `/pct_path2` |
| `rsc/tomogram/scene_map_sparse.pickle` | Output sparse map |
| `planner/build.sh` | Build C++ pybind modules |
| `rsc/tomogram/publish_sparse_tomogram.py` | Optional PointCloud2 publisher for the sparse nodes |
| `tomography/scripts/extract_largest_physical_component.py` | Offline sparse-pickle planner-reachability filter (keeps largest component) |
| `tomography/scripts/pcd_to_filtered_sparse.py` | PCD → sparse pickle → largest planner-reachable component (orchestrates the two scripts above) |

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

Filter the sparse tomogram to its largest planner-reachable component:

```bash
cd /home/nuc/numa/PctPlanner/tomography/scripts
python3 extract_largest_physical_component.py --pickle scene_map_sparse.pickle
# writes rsc/tomogram/scene_map_sparse_planner.pickle
```

One-shot PCD → filtered sparse pickle:

```bash
cd /home/nuc/numa/PctPlanner/tomography/scripts
python3 pcd_to_filtered_sparse.py --pcd dshp.pcd
# writes rsc/tomogram/scene_map_sparse_planner.pickle
```

Offline path planning on the filtered sparse map:

```bash
cd /home/nuc/numa/PctPlanner/planner/scripts
python3 Planner.py \
  --map scene_map_sparse_planner \
  --start -100 -10 0 \
  --goal -90 0 0 \
  --output /tmp/path.npy
```

Batched planning (many start/goal pairs in one C++ call):

```bash
python3 Planner.py \
  --map scene_map_sparse_planner \
  --json pairs.json --all-pairs --jobs 8 --output all_paths.json
```

`--all-pairs` routes through `SparseAstar::SearchBatch`: pairs are searched
concurrently inside C++ (each thread owns a `SearchScratch` indexed by
`node_id`; the graph `nodes_` is shared read-only — no processes or IPC).
`--jobs` is the C++ worker-thread count (default: `hardware_concurrency`).
Measured on `dshp_2` (7.75M nodes, 32 pairs incl. 9 disconnected failures):
29.5s serial → 6.7s at 16 threads (~4.4x). Failed searches can be the most
expensive (they explore the whole reachable component), so round-robin
schedules pairs across threads.

Batched planning notes:
- Live progress: as each pair finishes, the C++ layer prints a
  `[i/N] waypoints=..., length=... m` (or `Planning failed`) line immediately
  (printf + fflush, so it streams even when stdout is piped). Threaded runs
  print in completion order, not pair order. The C++ length is computed from
  grid diffs scaled by `resolution_` and node heights, so it can differ from
  the Python `path_length` in the last ~0.001 m due to float-accumulation
  order.
- Ctrl+C is interruptible: `search_batch` releases the Python GIL while the
  batch runs and polls `PyErr_CheckSignals` from the wait loop; on SIGINT it
  sets a cancel flag that workers check per-expansion, so a batch aborts
  within ~1 s (exit 130, nothing written). Workers read the map graph
  read-only, so releasing the GIL is safe.
- Deterministic across thread counts (same `--jobs` 1 and 16 produce identical
  output).
- A* open-set tie-breaking differs from single-pair `Search`, so a batch path
  can differ slightly from the one-at-a-time path (both valid; cost difference
  is ~0.01% on measured cases).
- `SparseAstar::Search` only calls `Reset()` when the previous search
  *succeeded* (`if (!result_.empty())`), so consecutive failed searches leave
  stale node `g/f/parent` that corrupt the next search. The batch scratch
  resets per search and does not have this bug.

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
  - `SPARSE_GATEWAY_COST_DELTA = 8.0`
  - `SPARSE_GATEWAY_HEIGHT_DELTA = 0.1`
- The sparse export does not perform secondary dilation or downsampling.
- The `robot_height_min` / ceiling-clearance check has been removed from the
  sparse mask at user request; nodes are filtered only by finite ground height,
  finite cost, and `trav <= SPARSE_ASTAR_COST_THRESHOLD` (plus forced gateways).
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
- [ ] `Planner.py --all-pairs --jobs N` produces the same records as
      `--jobs 1`, and its paths start/end at the snapped node world poses.
- [ ] `--all-pairs` streams a `[i/N] waypoints=..., length=... m` line per pair
      while the batch runs (visible mid-run in a live `tail`), then a
      `Finished: ...` summary.
- [ ] Ctrl+C during `--all-pairs` aborts within ~1 s (exit 130, prints
      `Interrupted: batch canceled ...`), and a no-interrupt run still writes
      the output JSON.
