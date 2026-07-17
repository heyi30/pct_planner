# PctPlanner

English | [中文](README.zh-CN.md)

PctPlanner is a ROS 2 / Python planning workspace for building sparse tomogram
maps from PCD files and planning 3-D paths with a pybind11 Sparse A* module.

The repository has been cleaned so generated data and large artifacts are not
tracked. Point clouds, tomogram pickles, build outputs, third-party source trees,
and Python caches must be kept local or regenerated.

## Environment

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- CUDA 12.8
- CuPy built for CUDA 12.x
- Open3D
- CMake and a C++ compiler

Source ROS before running ROS nodes:

```bash
source /opt/ros/humble/setup.bash
```

## Repository Layout

```text
pct_planner_msgs/      ROS 2 service definition for path requests
planner/lib/           Sparse A* C++/pybind source and build scripts
planner/scripts/       Offline planner CLI and CSV path publisher
tomography/scripts/    PCD to sparse tomogram generation/filtering tools
tomography/config/     Tomography and scene parameters
rsc/pcd/               Local PCD input files, ignored by Git
rsc/tomogram/          Local sparse tomogram pickle files, ignored by Git
rsc/rviz/              RViz config
```

## Ignored Local Artifacts

The following files and directories are local artifacts and should not be
committed:

- `*.pcd`
- `*.pickle`
- `*.bak`
- `*.npy`
- `*.pyc`
- `.DS_Store`
- `__pycache__/`
- `build/`, `install/`, `log/`
- `planner/lib/3rdparty/`
- `planner/lib/*.so`
- `rsc/traj/`

If a fresh clone is missing map data, copy the required PCD files into
`rsc/pcd/` and regenerate the tomogram pickle.

## Build

Build the Sparse A* pybind module:

```bash
cd /home/nuc/numa/PctPlanner/planner
./build.sh
```

The build places the Python extension in `planner/lib/`, for example:

```text
planner/lib/sparse_a_star.cpython-310-x86_64-linux-gnu.so
```

Set the module path before running planner scripts:

```bash
export PYTHONPATH=/home/nuc/numa/PctPlanner/planner/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib:$LD_LIBRARY_PATH
```

## Generate a Sparse Tomogram

Put an input PCD file under `rsc/pcd/`, then run:

```bash
cd /home/nuc/numa/PctPlanner/tomography/scripts
python3 pcd_to_filtered_sparse.py --pcd your_map.pcd
```

This script runs:

1. `tomography.py`, which exports `rsc/tomogram/scene_map_sparse.pickle`
2. `extract_largest_physical_component.py`, which keeps the largest reachable
   component and writes `rsc/tomogram/scene_map_sparse_planner.pickle`

`scene_map_sparse_planner.pickle` is the default map used by
`planner/scripts/Planner.py`.

Useful options:

```bash
python3 pcd_to_filtered_sparse.py \
  --pcd your_map.pcd \
  --output scene_map_sparse_planner.pickle \
```

## RViz Visualization

Publish the filtered sparse tomogram:

```bash
source /opt/ros/humble/setup.bash
cd /home/nuc/numa/PctPlanner/rsc/tomogram
python3 publish_sparse_tomogram.py
```

In RViz:

- Fixed Frame: `map`
- Display: `PointCloud2`
- Topic: `/sparse_tomogram`
- Color field: `trav` or `gateway`

Publish a raw PCD as `/global_points`:

```bash
source /opt/ros/humble/setup.bash
cd /home/nuc/numa/PctPlanner
python3 rsc/pcd/pcd_publisher.py
```

`pcd_publisher.py` currently loads `rsc/pcd/dshp.pcd`.

## Offline Path Planning

Run the standalone planner on the filtered sparse tomogram:

```bash
cd /home/nuc/numa/PctPlanner/planner/scripts
python3 Planner.py \
  --map scene_map_sparse_planner \
  --start -100 -10 0 \
  --goal -90 0 0 \
  --output path.csv
```

Arguments:

- `--map`: pickle basename under `rsc/tomogram/`, without `.pickle`
- `--start`: start position as `x y z`
- `--goal`: goal position as `x y z`
- `--output`: optional output file; supports `.csv`, `.npy`, `.pickle`, `.pcd`,
  or text

On success, the script prints snapped grid indices, path waypoint count, and
path length. `path.csv` contains `x,y,z` columns and can be published to ROS 2.

## Publish a CSV Path to ROS 2

`planner/scripts/pub_path.py` reads `planner/scripts/path.csv` by default and
publishes it as `nav_msgs/msg/Path`.

```bash
source /opt/ros/humble/setup.bash
cd /home/nuc/numa/PctPlanner
python3 planner/scripts/pub_path.py
```

Defaults:

- CSV: `planner/scripts/path.csv`
- Topic: `/pct_path`
- Frame: `map`
- Publish rate: `1.0 Hz`

Explicit options:

```bash
python3 planner/scripts/pub_path.py \
  --csv planner/scripts/path.csv \
  --topic /pct_path \
  --frame-id map \
  --rate 1.0
```

RViz can display the result with a `Path` display on `/pct_path`.

## Sparse Planning Parameters

These values must stay consistent between tomography export and Sparse A*:

```python
SPARSE_ASTAR_COST_THRESHOLD = 35.0
SPARSE_GATEWAY_COST_DELTA = 8.0
SPARSE_GATEWAY_HEIGHT_DELTA = 0.1
SPARSE_STEP_MAX = 0.5
SPARSE_COST_WEIGHT = 0.2
```

Relevant files:

- `tomography/scripts/tomography.py`
- `planner/scripts/sparse_planner_wrapper.py`
- `tomography/scripts/extract_largest_physical_component.py`
