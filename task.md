# 稀疏 Tomogram 地图与 Sparse A* 规划改造方案

## 目标

将 `tomography/scripts/tomography.py` 生成的 pickle 规划地图改为稀疏地图格式，并将规划侧改为只使用稀疏地图和 Sparse A* 搜索。

这是测试用点云：/home/nuc/numa/PctPlanner/rsc/pcd/dshp_downsampled.pcd

明确约束：

- 不保留 dense fallback。
- 不使用 dense A*。
- 不进行轨迹优化。
- 不调用 `DenseElevationMap`。
- 不调用 `GPMPOptimizer` 或 `GenerateTrajectory()`。
- 稀疏化阶段不做二次膨胀，因为 `Tomogram.point2map()` 输出的 `layers_t` 已经是 `inflated_cost`。

## 稀疏地图格式

输出文件建议固定为：

```text
rsc/tomogram/scene_map_sparse.pickle
```

pickle 内容固定为：

```python
{
    "format": "tomogram_sparse_v1",
    "shape": [n_slice, map_dim_y, map_dim_x],
    "resolution": resolution,
    "center": center,
    "slice_h0": slice_h0,
    "slice_dh": slice_dh,
    "slice_heights": slice_heights,

    "indices": np.ndarray[N, 3],   # int32, [layer, row, col]
    "trav": np.ndarray[N],         # float32
    "elev_g": np.ndarray[N],       # float32
    "elev_c": np.ndarray[N],       # float32
    "gateway": np.ndarray[N],      # int8 or int32, 0 / 2 / -2
}
```

要求：

- 不保存 dense `data` 字段。
- `indices` 是唯一的规划节点来源。
- `trav/elev_g/elev_c/gateway` 与 `indices` 一一对应。
- 数值字段用于规划时建议保存为 `float32`，不要用 `float16` 作为规划输入。

## 稀疏化规则

`Tomogram.point2map()` 当前返回的 `layers_t` 来自：

```python
layers_t = self.inflated_cost[idx_simp].get()
```

因此 dense tomogram 已经完成障碍/代价膨胀，稀疏化阶段只做节点筛选和编码，不再改变安全边界。

稀疏节点集合：

```python
valid_height = np.isfinite(layers_g)
valid_cost = np.isfinite(layers_t)

clearance_ok = (
    np.isfinite(layers_c)
    & ((layers_c - layers_g) >= robot_height_min)
)

traversable = layers_t <= astar_cost_threshold
gateway_keep = gateway != 0

sparse_mask = (
    valid_height
    & valid_cost
    & clearance_ok
    & traversable
) | gateway_keep
```

关键要求：

- `astar_cost_threshold` 必须和 Sparse A* 搜索阈值一致。
- `gateway_keep` 强制保留，避免跨层拓扑被删。
- 不做 `binary_dilation`。
- 不做降采样。
- 不修改 `layers_t`。
- 如果后续发现路径断连，优先检查阈值和 gateway，而不是在稀疏化阶段增加膨胀。

## Gateway 生成

沿用当前 `planner/scripts/planner_wrapper.py` 中 dense planner 的 gateway 逻辑，并移动到稀疏地图生成或稀疏 planner 初始化阶段。

```python
diff_t = trav[1:] - trav[:-1]
diff_g = np.abs(elev_g[1:] - elev_g[:-1])

gateway_up = np.zeros_like(trav, dtype=bool)
mask_t = diff_t < -gateway_cost_delta
mask_g = (diff_g < gateway_height_delta) & np.isfinite(elev_g[1:])
gateway_up[:-1] = mask_t & mask_g

gateway_dn = np.zeros_like(trav, dtype=bool)
mask_t = diff_t > gateway_cost_delta
mask_g = (diff_g < gateway_height_delta) & np.isfinite(elev_g[:-1])
gateway_dn[1:] = mask_t & mask_g

gateway = np.zeros_like(trav, dtype=np.int32)
gateway[gateway_up] = 2
gateway[gateway_dn] = -2
```

建议配置项：

```python
gateway_cost_delta = 8.0
gateway_height_delta = 0.1
```

## Sparse A* 数据结构

C++ 侧新增 `SparseAstar`，不再构建完整 `MultiLayerGridMap`。

核心节点结构：

```cpp
struct SparseNode {
  Eigen::Vector3i idx;   // layer, row, col
  double cost;
  double height;
  double ceiling;
  int gateway;
  double g = 1e9;
  double f = 1e9;
  SparseNode* parent = nullptr;
};
```

核心容器：

```cpp
std::unordered_map<int, SparseNode> nodes_;
```

hash 规则：

```cpp
hash = layer * max_y_ * max_x_ + row * max_x_ + col;
```

所有路径节点必须来自 `nodes_`，不存在于 `nodes_` 的 cell 视为不可走。

## 邻接关系

同层 8 邻域连接：

```text
(layer, row, col) -> (layer, row + dr, col + dc)
dr, dc in 8-neighborhood
```

同层边允许条件：

```text
neighbor exists in sparse node set
abs(neighbor.height - current.height) <= step_max
neighbor.cost <= astar_cost_threshold
```

跨层连接只允许通过 gateway：

```text
gateway > 0: 尝试 layer + 1
gateway < 0: 尝试 layer - 1
```

跨层目标第一版限定为同 `row,col`。如果实际地图层间存在轻微错位，再扩展为目标层 8 邻域内最近节点。

跨层边要求：

```text
target layer in range
target sparse node exists
height difference satisfies configured limit
target node cost <= astar_cost_threshold or target gateway != 0
```

边权：

```text
distance = sqrt(dx^2 + dy^2 + dz^2)
cost_penalty = cost_weight * neighbor.cost
edge_cost = distance + cost_penalty
```

`dx/dy/dz` 可以先使用 grid 尺度；如果使用米制，则统一乘 `resolution`。

## 起终点处理

输入 `start_pos/end_pos` 先转换为 grid index：

```python
col = round((x - center_x) / resolution) + map_dim_x / 2
row = round((y - center_y) / resolution) + map_dim_y / 2
layer = z2slice(z)
```

随后必须吸附到稀疏节点：

```text
同层优先
半径从 0 到 max_snap_radius 逐圈搜索
候选节点必须存在于 sparse node set
候选节点 cost <= astar_cost_threshold
选 grid 距离最近的节点
距离相同则选 cost 更低的节点
```

如果找不到：

```text
直接规划失败
不 fallback 到 dense
不临时插入节点
```

日志必须输出：

```text
raw_start_idx
snapped_start_idx
start_snap_distance
raw_goal_idx
snapped_goal_idx
goal_snap_distance
```

## Python 规划接口

建议新增轻量规划类：

```python
SparseTomogramPlanner
```

或者将现有 `TomogramPlanner` 改为只支持稀疏地图。

加载要求：

```python
if data_dict.get("format") != "tomogram_sparse_v1":
    raise ValueError("expected tomogram_sparse_v1")
```

不允许自动转换旧 dense pickle。

规划流程固定为：

```text
load sparse pickle
build sparse hash table
convert start/end to raw grid index
snap start/end to sparse nodes
SparseAstar.search(start_node, goal_node)
path_grid = SparseAstar.get_result_matrix()
path_xyz = sparse_path_to_map_path(path_grid)
return or publish path_xyz
```

路径坐标转换：

```python
x = (col - offset_x) * resolution + center_x
y = (row - offset_y) * resolution + center_y
z = node.elev_g
```

明确禁止调用：

```text
OfflineElePlanner.init_map
DenseElevationMap.Init
GPMPOptimizer
GenerateTrajectory
```

## ROS 输出

规划输出为 Sparse A* 原始折线路径。

如果发布 `nav_msgs/Path`：

- `pose.position.x = x`
- `pose.position.y = y`
- `pose.position.z = z`

如果现有控制器仍期望 `np.ndarray[N, 3]`：

- 继续返回 `path_xyz`
- 内容为未优化的 Sparse A* 折线路径
- 不做平滑优化

## 实施步骤

1. 清理 `tomography/scripts/tomography.py`
   - 删除重复的 `process/exportTomogram/publishTomogram` 定义。
   - 保留一套实际生效的处理流程。

2. 在 `tomography.py` 中新增 sparse export
   - 计算 gateway。
   - 计算 `sparse_mask`。
   - 从 dense 临时数组抽取 `indices/trav/elev_g/elev_c/gateway`。
   - 导出 `scene_map_sparse.pickle`。

3. 增加 sparse map 日志
   - dense cell 数。
   - sparse node 数。
   - sparse ratio。
   - 文件大小。
   - 导出耗时。

4. 新增 C++ `SparseAstar`
   - 实现 sparse node 初始化。
   - 实现 hash 查询。
   - 实现同层 8 邻域。
   - 实现 gateway 跨层邻接。
   - 实现 search/result/visited 输出。

5. 增加 pybind 接口
   - 暴露 `SparseAstar`。
   - 暴露 `init/search/get_result_matrix/get_visited_set`。

6. 改造 Python planner wrapper
   - 只接受 `tomogram_sparse_v1`。
   - 初始化 Sparse A*。
   - 实现起终点吸附。
   - 返回 Sparse A* 原始路径。

7. 改造 ROS 规划脚本
   - 加载 `scene_map_sparse.pickle`。
   - 使用 sparse planner。
   - 发布 Sparse A* path。
   - 移除轨迹优化相关调用。

8. 编译与验证
   - 重新构建 planner pybind 模块。
   - 运行 PCD 到 sparse pickle 生成。
   - 运行 sparse planner。
   - RViz 检查路径。

## 可验证完成指标

### 1. 地图文件指标

- 运行 `tomography.py` 后生成：

```text
rsc/tomogram/scene_map_sparse.pickle
```

- pickle 中：

```python
data_dict["format"] == "tomogram_sparse_v1"
"data" not in data_dict
indices.shape == (N, 3)
len(trav) == len(elev_g) == len(elev_c) == len(gateway) == N
```

### 2. 稀疏率指标

日志必须打印：

```text
dense_cells = n_slice * map_dim_y * map_dim_x
sparse_nodes = N
sparse_ratio = N / dense_cells
```

验收条件：

```text
sparse_nodes < dense_cells
sparse_ratio < 0.6
```

`sparse_ratio < 0.6` 是初始目标，可根据实际场景调整。

### 3. 安全边界指标

稀疏化阶段不得出现任何二次膨胀调用。

所有非 gateway 节点必须满足：

```python
trav <= astar_cost_threshold
np.isfinite(elev_g)
elev_c - elev_g >= robot_height_min
```

并且：

```text
tomography sparse export threshold == Sparse A* search threshold
```

### 4. 加载失败指标

规划侧加载旧 dense pickle 时必须失败，并输出类似：

```text
expected tomogram_sparse_v1
```

禁止行为：

- 自动转换 dense。
- fallback 到 dense planner。
- fallback 到旧 `OfflineElePlanner`。

### 5. 搜索路径指标

Sparse A* 返回的每个 path node 都必须存在于 sparse node hash table。

相邻 path node 必须满足：

- 同层移动：8 邻域关系。
- 跨层移动：至少一端 `gateway != 0`。

path 中非 gateway 节点必须满足：

```text
cost <= astar_cost_threshold
```

### 6. 起终点吸附指标

日志必须打印：

```text
raw_start_idx
snapped_start_idx
start_snap_distance
raw_goal_idx
snapped_goal_idx
goal_snap_distance
```

验收条件：

```text
start_snap_distance <= max_snap_radius
goal_snap_distance <= max_snap_radius
```

如果超过半径找不到 sparse node：

```text
规划失败
不 fallback
不临时插入节点
```

### 7. 禁用轨迹优化指标

运行规划时不得出现以下调用或日志：

```text
GenerateTrajectory
DenseElevationMap
OfflineElePlanner.init_map
GPMPOptimizer
```

验收条件：

- 返回路径是 Sparse A* 原始折线。
- RViz 发布路径点数量等于 Sparse A* path 节点数，除非只做明确的轻量抽稀。
- 不做优化平滑。

### 8. 性能指标

日志必须打印：

```text
sparse_astar_init_ms
sparse_astar_search_ms
visited_nodes
path_nodes
```

验收条件：

```text
visited_nodes <= sparse_nodes
path_nodes > 0 for reachable start/goal
```

大地图上应满足：

- 初始化内存显著低于 dense grid map。
- 搜索耗时不高于现有 dense A*。

如果耗时高于 dense A*，优先检查：

- hash 查询实现。
- 起终点吸附半径。
- gateway 邻居生成。
- 是否保留了过多高 cost 边界节点。

### 9. 端到端指标

完整链路必须可运行：

```text
PCD -> scene_map_sparse.pickle -> Sparse A* -> path_xyz / ROS Path -> RViz
```

验收条件：

- RViz 中路径起点和终点位置正确。
- 路径 z 值来自 sparse node 的 `elev_g`。
- 路径连续，无无效跳点。
- 运行过程中没有 dense map 规划依赖。
