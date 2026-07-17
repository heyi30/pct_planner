# PctPlanner

[English](README.md) | 中文

PctPlanner 是一个基于 ROS 2 / Python 的路径规划工作空间，用于从 PCD
点云生成稀疏 tomogram 地图，并通过 pybind11 Sparse A* 模块规划三维路径。


## 环境要求

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- CUDA 12.8
- 匹配 CUDA 12.x 的 CuPy
- Open3D
- CMake 和 C++ 编译器

运行 ROS 节点前先加载 ROS 环境：

```bash
source /opt/ros/humble/setup.bash
```

## 目录结构

```text
pct_planner_msgs/      ROS 2 路径规划服务定义
planner/lib/           Sparse A* C++ / pybind 源码和构建脚本
planner/scripts/       离线规划 CLI 和 CSV 路径发布脚本
tomography/scripts/    PCD 到稀疏 tomogram 的生成和过滤工具
tomography/config/     Tomography 与场景参数
rsc/pcd/               本地点云输入文件，Git 忽略
rsc/tomogram/          本地稀疏 tomogram pickle 文件，Git 忽略
rsc/rviz/              RViz 配置
```

## 构建

构建 Sparse A* pybind 模块：

```bash
cd /home/nuc/numa/PctPlanner/planner
./build.sh
```

构建后会在 `planner/lib/` 下生成 Python 扩展，例如：

```text
planner/lib/sparse_a_star.cpython-310-x86_64-linux-gnu.so
```

运行 planner 脚本前设置模块路径：

```bash
export PYTHONPATH=/home/nuc/numa/PctPlanner/planner/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/home/nuc/numa/PctPlanner/planner/lib:$LD_LIBRARY_PATH
```

## 生成稀疏 Tomogram

先把输入 PCD 文件放到 `rsc/pcd/`，然后运行：

```bash
cd /home/nuc/numa/PctPlanner/tomography/scripts
python3 pcd_to_filtered_sparse.py --pcd your_map.pcd
```

这个脚本会依次执行：

1. `tomography.py`：生成 `rsc/tomogram/scene_map_sparse.pickle`
2. `extract_largest_physical_component.py`：过滤最大可达连通区域，并生成
   `rsc/tomogram/scene_map_sparse_planner.pickle`

`scene_map_sparse_planner.pickle` 是 `planner/scripts/Planner.py` 默认使用的地图。

常用参数示例：

```bash
python3 pcd_to_filtered_sparse.py \
  --pcd your_map.pcd \
  --output scene_map_sparse_planner.pickle \
```

## RViz 可视化

发布过滤后的稀疏 tomogram：

```bash
source /opt/ros/humble/setup.bash
cd /home/nuc/numa/PctPlanner/rsc/tomogram
python3 publish_sparse_tomogram.py
```

RViz 中添加：

- Fixed Frame：`map`
- Display：`PointCloud2`
- Topic：`/sparse_tomogram`
- 颜色字段：`trav` 或 `gateway`

发布原始 PCD 到 `/global_points`：

```bash
source /opt/ros/humble/setup.bash
cd /home/nuc/numa/PctPlanner
python3 rsc/pcd/pcd_publisher.py
```

当前 `pcd_publisher.py` 固定读取 `rsc/pcd/dshp.pcd`。

## 离线路径规划

使用过滤后的稀疏 tomogram 执行离线规划：

```bash
cd /home/nuc/numa/PctPlanner/planner/scripts
python3 Planner.py \
  --map scene_map_sparse_planner \
  --start -100 -10 0 \
  --goal -90 0 0 \
  --output path.csv
```

参数说明：

- `--map`：`rsc/tomogram/` 下的 pickle 文件名，不带 `.pickle`
- `--start`：起点坐标，格式为 `x y z`
- `--goal`：终点坐标，格式为 `x y z`
- `--output`：可选输出文件，支持 `.csv`、`.npy`、`.pickle`、`.pcd` 或文本

规划成功后，脚本会打印起终点吸附后的网格索引、路径点数量和路径长度。
`path.csv` 使用 `x,y,z` 三列，可继续发布到 ROS 2。

## 发布 CSV 路径到 ROS 2

`planner/scripts/pub_path.py` 默认读取 `planner/scripts/path.csv`，并发布为
`nav_msgs/msg/Path`。

```bash
source /opt/ros/humble/setup.bash
cd /home/nuc/numa/PctPlanner
python3 planner/scripts/pub_path.py
```

默认参数：

- CSV：`planner/scripts/path.csv`
- Topic：`/pct_path`
- Frame：`map`
- 发布频率：`1.0 Hz`

显式指定参数：

```bash
python3 planner/scripts/pub_path.py \
  --csv planner/scripts/path.csv \
  --topic /pct_path \
  --frame-id map \
  --rate 1.0
```

RViz 中添加 `Path` display，并订阅 `/pct_path` 即可查看路径。

## 稀疏规划参数

以下参数需要在 tomography 导出和 Sparse A* 规划侧保持一致：

```python
SPARSE_ASTAR_COST_THRESHOLD = 35.0
SPARSE_GATEWAY_COST_DELTA = 8.0
SPARSE_GATEWAY_HEIGHT_DELTA = 0.1
SPARSE_STEP_MAX = 0.5
SPARSE_COST_WEIGHT = 0.2
```

相关文件：

- `tomography/scripts/tomography.py`
- `planner/scripts/sparse_planner_wrapper.py`
- `tomography/scripts/extract_largest_physical_component.py`