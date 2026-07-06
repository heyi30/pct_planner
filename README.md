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

### Tomogram Construction

- In **tomography/scripts/**, run **tomography.py** :

```bash
cd tomography/scripts/
python3 tomography.py
```

- The generated tomogram is visualized as ROS PointCloud2 message in RViz and saved in **rsc/tomogram/**.

### Trajectory Generation

- In **planner/scripts/**, run **plan.py** with the **--scene** argument:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/YOUR/DIRECTORY/TO/PCT_planner/planner/lib/3rdparty/gtsam-4.1.1/install/lib
cd planner/scripts/
export LD_LIBRARY_PATH=/home/unitree/navigation/PctPlanner/planner/lib/3rdparty/gtsam-4.1.1/install/lib:$LD_LIBRARY_PATH
python3 plan.py
```

- The generated trajectory is visualized as ROS Path message in RViz.

### System-based Trajectory Generation (`plan_systemt.py`)

`plan_systemt.py` 使用系统位姿和里程计信息自动规划路径：

- 启动时，从 `rsc/tomogram/scene_map.pickle` 加载 tomogram 数据并初始化规划器。
- 初始起点默认为 `(0, 0, 0)`，终点默认为 `(1, 6, 0)`。
- 订阅 `/odom/ground_truth` (`nav_msgs/Odometry`)，每次收到最新位姿后，从**当前位姿**到终点重新规划路径。
- 当当前位姿与终点的平面距离小于 2 m 时，不再重新规划，只按固定频率（默认 1 Hz）发布当前路径到 `/pct_path_system`。

使用方法：

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/YOUR/DIRECTORY/TO/PCT_planner/planner/lib/3rdparty/gtsam-4.1.1/install/lib
cd planner/scripts/
python3 plan_systemt.py
```

> 在运行前，请确保：
>
> - `rsc/tomogram/scene_map.pickle` 已存在（可通过 `tomography.py` 生成），
> - 有节点在 `/odom/ground_truth` 上发布 `nav_msgs/Odometry` 消息，
> - 在 RViz 中订阅 `/pct_path_system` 可查看路径。

**修改初始起点和终点：**

在 `planner/scripts/plan_systemt.py` 中找到 `__init__` 里的以下两行：

```python
self.start_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 初始起点
self.goal_pos  = np.array([1.0, 6.0, 0.0], dtype=np.float32)  # 固定终点
```

- 如需修改**初始起点**，直接改 `self.start_pos` 中的三个数值，例如：

  ```python
  self.start_pos = np.array([x0, y0, z0], dtype=np.float32)
  ```
- 如需修改**终点**，直接改 `self.goal_pos` 中的三个数值，例如：

  ```python
  self.goal_pos = np.array([xg, yg, zg], dtype=np.float32)
  ```

注意：

- `start_pos` 只是**初始起点**，之后会被 `/odom/ground_truth` 的最新位姿覆盖，用于重新规划；
- `goal_pos` 是**固定目标点**，只有在你修改代码并重新运行 `plan_systemt.py` 后才会改变。

# 极简化运行

1.将planner/scripts/plan_direct.py文件中的tomogram_path = "/home/hanjiatong/PctPlanner/rsc/tomogram/scene_map.pickle"改为自己相对应的文件路径
2.运行rsc文件夹下的publish_start_end_pos.py
3.运行plan_direct.py
