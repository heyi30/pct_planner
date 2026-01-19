import os
import sys
import pickle
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import ByteMultiArray
from geometry_msgs.msg import Point, PoseStamped

from utils import *
from planner_wrapper import TomogramPlanner

sys.path.append('../')
from config import Config


def save_traj_as_pcd(traj, filename):
    """Saves a trajectory to a .pcd file."""
    with open(filename, 'w') as f:
        f.write('# .PCD v.7 - Point Cloud Data file format\n')
        f.write('VERSION .7\n')
        f.write('FIELDS x y z\n')
        f.write('SIZE 4 4 4\n')
        f.write('TYPE F F F\n')
        f.write('COUNT 1 1 1\n')
        f.write(f'WIDTH {len(traj)}\n')
        f.write('HEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write(f'POINTS {len(traj)}\n')
        f.write('DATA ascii\n')
        for point in traj:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')
def reverse_trajectory(traj):
    """Reverses a trajectory (flips the order of points).
    
    Args:
        traj: numpy array of shape (N, 3)
        
    Returns:
        reversed numpy array of shape (N, 3)
    """
    return traj[::-1]

def save_traj_to_csv(traj, filename):
    """Saves a trajectory to a CSV file."""
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z'])  # Header
        for point in traj:
            writer.writerow([point[0], point[1], point[2]])


def load_traj_from_csv(filename):
    """Loads a trajectory from a CSV file.
    
    Returns:
        numpy array of shape (N, 3) where N is the number of points
    """
    import csv
    points = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            points.append([float(row[0]), float(row[1]), float(row[2])])
    return np.array(points, dtype=np.float32)
class PCTPlannerSystemNode(Node):
    def __init__(self):
        super().__init__('pct_planner_system')

        self.cfg = Config()
        self.planner = TomogramPlanner(self.cfg)

        # 定义两个 tomogram 路径
        # tomogram_path1 = "/home/hanjiatong/PctPlanner/rsc/tomogram/scene_map1.pickle"
        # tomogram_path2 = "/home/hanjiatong/PctPlanner/rsc/tomogram/scene_map2.pickle"
        tomogram_path1 = "/home/unitree/navigation/PctPlanner/rsc/tomogram/scene_map_v3_narrow.pickle"
        tomogram_path2 = "/home/unitree/navigation/PctPlanner/rsc/tomogram/scene_map_v3_wide.pickle"
        traj = load_traj_from_csv("/home/unitree/navigation/PctPlanner/rsc/traj/cabin_path.csv")
        self.fixed_traj = reverse_trajectory(traj)
        self.fixed_traj2 = load_traj_from_csv("/home/unitree/navigation/PctPlanner/rsc/traj/return_path.csv")
        # 定义矩形区域 (x_min, x_max, y_min, y_max)
        # 当机器人在此区域内时使用 costmap1，否则使用 costmap2
        self.region_x_min = -38.5
        self.region_x_max = -28.3
        self.region_y_min = 4.5
        self.region_y_max = 10.7
        
        self.get_logger().info(f"Defined rectangle region: x=[{self.region_x_min}, {self.region_x_max}], y=[{self.region_y_min}, {self.region_y_max}]")
        
        # 加载第一个 tomogram
        self.get_logger().info(f"Loading tomogram1 from: {tomogram_path1}")
        try:
            with open(tomogram_path1, 'rb') as f:
                self.data_dict1 = pickle.load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load tomogram1 pickle: {e}")
            raise
        
        # 加载第二个 tomogram
        self.get_logger().info(f"Loading tomogram2 from: {tomogram_path2}")
        try:
            with open(tomogram_path2, 'rb') as f:
                self.data_dict2 = pickle.load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load tomogram2 pickle: {e}")
            raise
        
        # 初始化时先用 fixed_traj
        self.current_traj =self.fixed_traj
        self.should_plan = False
        pickled_data = pickle.dumps(self.data_dict2)
        msg = ByteMultiArray()
        msg.data = [bytes([b]) for b in pickled_data]
        
        self.get_logger().info("Initializing planner with costmap1...")
        self.planner.update_tomogram_from_msg(msg)
        self.tomogram_received = True
        self.current_costmap = 1  # 记录当前使用的 costmap
        self.get_logger().info("Planner initialized with costmap1.")

        # 起点位姿：初始为 (0, 0, 0)
        self.start_pos = np.array([-31.7,7.07,2.1], dtype=np.float32)
        # # 目标位姿：固定为 (1, 6, 0)
        # self.start_pos = np.array([39.2952, -5.03867,0.25], dtype=np.float32)
        self.goal_pos2 = np.array([-43.56449939946788,-14.250800335906225,0.0], dtype=np.float32)
        self.goal_pos1 = np.array([-39.0877, 3.27937, 0], dtype=np.float32)
        self.goal_pos = self.goal_pos2

        self.path_pub = self.create_publisher(Path, '/pct_path2', 10)

        # 当前轨迹缓存，用于定频率发布
        # self.current_traj = None
        # 1 Hz 定时器，按固定频率发布路径
        self.publish_timer = self.create_timer(1.0, self.publish_loop)

        # 订阅 /local_pose (geometry_msgs/PoseStamped)，随时更新 start_pos 并重新规划
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/local_pose',
            self.odom_callback,
            10,
        )

        self.get_logger().info('Initial start pose is (0, 0, 0).')
        self.get_logger().info('Goal pose is fixed at (1, 6, 0).')

        # 初始化完成后，立刻尝试用原点到目标做一次规划
        self.try_plan()

    def is_inside_region(self, x, y):
        """检查位置 (x, y) 是否在定义的矩形区域内。"""
        return (self.region_x_min <= x <= self.region_x_max and 
                self.region_y_min <= y <= self.region_y_max)
    
    def update_costmap_based_on_position(self, x, y):
        # """根据当前位置更新 costmap。如果在区域内使用 costmap1，否则使用 costmap2。"""
        # inside_region = self.is_inside_region(x, y)
        # desired_costmap = 1 if inside_region else 2
        
        # # 只在需要切换时才更新
        # if desired_costmap != self.current_costmap:
        #     if desired_costmap == 1:
        #         self.get_logger().info(f'Switching to costmap1 (inside region)')
        #         pickled_data = pickle.dumps(self.data_dict1)
        #         self.goal_pos = self.goal_pos1
        #     else:
        #         self.get_logger().info(f'Switching to costmap2 (outside region)')
        #         self.goal_pose = self.goal_pos2
        #         pickled_data = pickle.dumps(self.data_dict2)
            
        #     msg = ByteMultiArray()
        #     msg.data = [bytes([b]) for b in pickled_data]
        #     self.planner.update_tomogram_from_msg(msg)
        #     self.current_costmap = desired_costmap
        inside_region = self.is_inside_region(x, y)
        desired_costmap = 1 if inside_region else 2
        
        # 只在需要切换时才更新
        if desired_costmap != self.current_costmap:
            if desired_costmap == 1:
                self.get_logger().info(f'Switching to fixed path (inside region)')
                pickled_data = pickle.dumps(self.data_dict1)
                self.goal_pos = self.goal_pos1
                self.should_plan = False
                self.current_traj = self.fixed_traj
            else:
                self.get_logger().info(f'Switching to costmap2 (outside region)')
                self.goal_pos = self.goal_pos2
                pickled_data = pickle.dumps(self.data_dict2)
                self.should_plan = False
                self.current_traj = self.fixed_traj2
            
            msg = ByteMultiArray()
            msg.data = [bytes([b]) for b in pickled_data]
            self.planner.update_tomogram_from_msg(msg)
            self.current_costmap = desired_costmap
            self.try_plan()

    def odom_callback(self, msg: PoseStamped):
        """接收 /local_pose (PoseStamped)，更新当前起点，并重新规划。"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.start_pos = np.array([x, y, z], dtype=np.float32)
        self.get_logger().info(f'Received odom pose as new start: {self.start_pos}')
        
        # 根据当前位置更新 costmap
        self.update_costmap_based_on_position(x, y)
        start_pos_np = self.start_pos
        end_pos_np = self.goal_pos
        dist = np.linalg.norm(start_pos_np[:2] - end_pos_np[:2])
        if dist < 0.25:
            self.get_logger().info(
                'Reach Goal(<0.2m), terminate planner'
            )
            this.destroy_node()
        # 每次收到新里程计位姿，都重新规划一次
        self.try_plan()

    def try_plan(self):
        if not self.should_plan:
            return
        """在 tomogram 准备好时，从当前 start_pos 规划到 goal_pos。"""
        if not self.tomogram_received:
            self.get_logger().info('Tomogram not ready yet. Cannot plan path.')
            return

        start_pos_np = self.start_pos
        end_pos_np = self.goal_pos

        
        # 如果当前起点与目标距离小于 2m，则不再重新规划，只保留/发布当前轨迹
        dist = np.linalg.norm(start_pos_np[:2] - end_pos_np[:2])
        if dist < 2.0:
            self.get_logger().info(
                f'Distance to goal is {dist:.2f} m (< 2 m). Skip replanning, only publishing current trajectory.'
            )
            if dist < 0.25:
                self.get_logger().info(
                    'Reach goal(<0.2m), terminalte planner'
                )
                # raise SystemExit("reach target")
                this.destroy_node()
            return
        # if dist < 0.1:
        #     self.get_logger().info(
        #         'Reach goal(<0.2m), terminalte planner'
        #     )
        #     raise SystemExit("reach target")
        self.get_logger().info(
            f'Planning trajectory from start {start_pos_np.tolist()} to goal {end_pos_np.tolist()}...'
        )

        traj_3d = self.planner.plan(
            start_pos_np[:2],
            end_pos_np[:2],
            start_pos_np[2] + 0.5,
            end_pos_np[2] + 0.5,
        )

        if traj_3d is not None:
            # 更新当前轨迹，由定时器按固定频率发布
            self.current_traj = traj_3d
            # 如果你需要保存为 pcd，可以解除下面注释
            # save_traj_as_pcd(traj_3d, 'trajectory_system.pcd')
            self.get_logger().info('Trajectory planned. Will be published at 10 Hz to /pct_path_system.')
        else:
            self.get_logger().warn('Failed to generate a trajectory.')

    def publish_loop(self):
        """以固定频率（1 Hz）发布当前轨迹到 /pct_path_system。"""
        if self.current_traj is None:
            return

        path_msg = traj2ros(self.current_traj)
        self.path_pub.publish(path_msg)



def main(args=None):
    rclpy.init(args=args)

    node = PCTPlannerSystemNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
