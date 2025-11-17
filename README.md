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
python3 plan.py
```

- The generated trajectory is visualized as ROS Path message in RViz.

# 极简化运行
1.将planner/scripts/plan_direct.py文件中的tomogram_path = "/home/hanjiatong/PctPlanner/rsc/tomogram/scene_map.pickle"改为自己相对应的文件路径
2.运行rsc文件夹下的publish_start_end_pos.py
3.运行plan_direct.py