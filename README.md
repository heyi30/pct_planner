- Ubuntu == 22.04
- ROS == humble
- CUDA == 12.8

### Python

- Python == 3.10
- [CuPy](https://docs.cupy.dev/en/stable/install.html) with CUDA == 12.8
- Open3d

## Build & Install

In **planner/**, run **build_thirdparty.sh** first and then run **build.sh**.

```bash
cd planner/
./build_thirdparty.sh
./build.sh
```

The build produces the C++ pybind modules in `planner/lib/`, including the
Sparse A* module:

- `sparse_a_star.cpython-310-x86_64-linux-gnu.so`
- `libsparse_a_star_search.so`

Before running any planner script, set up the library and Python paths:

```bash
export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/nuc/numa/PctPlanner/planner/lib:$PYTHONPATH
```

## Tomogram Construction

In **tomography/scripts/**, run **tomography.py**:

```bash
cd tomography/scripts/
python3 tomography.py
```

- The generated **sparse** tomogram is saved as
  `rsc/tomogram/scene_map_sparse.pickle` in `tomogram_sparse_v1` format.
- The tomogram surface is also visualized as a ROS `PointCloud2` message in RViz on
  `/tomogram`.
- The legacy dense `scene_map.pickle` is no longer produced.

### Visualizing the sparse map

To visualize the actual sparse planning nodes (not just the surface point cloud),
run:

```bash
cd rsc/tomogram/
python3 publish_sparse_tomogram.py
```

In RViz:

- Add a **PointCloud2** display.
- Set the topic to `/sparse_tomogram`.
- Set the Fixed Frame to `map`.
- Color by the `trav` or `gateway` channel.

### Sparse map parameters

The following thresholds must stay in sync between the tomography export and
`SparseTomogramPlanner` / `SparseAstar`:

```python
SPARSE_ASTAR_COST_THRESHOLD = 35.0
SPARSE_ROBOT_HEIGHT_MIN     = 0.6
SPARSE_GATEWAY_COST_DELTA   = 8.0
SPARSE_GATEWAY_HEIGHT_DELTA = 0.1
```

## Trajectory Generation

The planner now uses **only** the sparse tomogram and Sparse A*. There is no
dense fallback, no `DenseElevationMap`, and no GPMP trajectory optimization.

- In **planner/scripts/**, run **plan.py**:

```bash
export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/nuc/numa/PctPlanner/planner/lib:$PYTHONPATH
cd planner/scripts/
python3 plan.py
```

`plan.py` loads `rsc/tomogram/scene_map_sparse.pickle` at startup, then waits
for start/end positions on `/start_pos` and `/end_pos`. The raw Sparse A*
polyline is published as a `nav_msgs/Path` on `/pct_path`.

## System-based Trajectory Generation (`plan_systemt.py`)

`plan_systemt.py` uses system pose and odometry information to plan repeatedly:

- 启动时，从 `rsc/tomogram/scene_map_sparse.pickle` 加载稀疏 tomogram 数据并初始化
  `SparseTomogramPlanner`。
- 默认起点为楼梯终点，默认终点为 lobby（已在代码中通过 `M_loc2pct` 转换到 tomogram
  坐标系）。
- 订阅 `/local_pose` (`geometry_msgs/PoseStamped`)，每次收到最新位姿后，从**当前位姿**
  到终点重新规划 Sparse A* 路径。
- 当当前位姿与终点的平面距离小于 2 m 时，不再重新规划，只按固定频率（默认 1 Hz）发布当前路径到 `/pct_path2`。

使用方法：

```bash
export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/nuc/numa/PctPlanner/planner/lib:$PYTHONPATH
cd planner/scripts/
python3 plan_systemt.py
```

> 在运行前，请确保：
>
> - `rsc/tomogram/scene_map_sparse.pickle` 已存在（可通过 `tomography.py` 生成），
> - 有节点在 `/local_pose` 上发布 `geometry_msgs/PoseStamped` 消息，
> - 在 RViz 中订阅 `/pct_path2` 可查看路径。

**修改起点和终点：**

在 `planner/scripts/plan_systemt.py` 的 `__init__` 中修改 `stair_end` 和 `lobby`
两个数组（local 坐标系），它们会自动通过 `M_loc2pct` 变换到 tomogram 坐标系。

## Minimal run (`plan_direct.py`)

1. Make sure `rsc/tomogram/scene_map_sparse.pickle` exists.
2. Run the start/end publisher:

   ```bash
   cd rsc/
   python3 publish_start_end_pos.py
   ```

3. Run the direct planner:

   ```bash
   export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib:$LD_LIBRARY_PATH
   export PYTHONPATH=/home/nuc/numa/PctPlanner/planner/lib:$PYTHONPATH
   cd planner/scripts/
   python3 plan_direct.py
   ```

The raw Sparse A* path is published as a `nav_msgs/Path` on `/pct_path2`.

## Verification

When planning succeeds, the console prints:

```text
raw_start_idx = [...]
snapped_start_idx = [...]
start_snap_distance = ...
raw_goal_idx = [...]
snapped_goal_idx = [...]
goal_snap_distance = ...
sparse_astar_init_ms = ...
sparse_astar_search_ms = ...
visited_nodes = ...
path_nodes = ...
```

In RViz the path should:

- Start and end at the requested positions,
- Use z values from the sparse node `elev_g`,
- Be an unoptimized polyline.
