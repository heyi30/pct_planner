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


class PCTPlannerSystemNode(Node):
    def __init__(self):
        super().__init__('pct_planner_system')

        self.cfg = Config()
        self.planner = TomogramPlanner(self.cfg)

        # 直接从 pickle 加载 tomogram，并初始化 planner
        #tomogram_path = "/home/hanjiatong/PctPlanner/rsc/tomogram/scene_map.pickle"
        tomogram_path = "/home/unitree/navigation/PctPlanner/rsc/tomogram/scene_map.pickle"
        # tomogram_path = "/home/ros/ros2_ws/PctPlanner/rsc/tomogram/scene_map.pickle"
        self.get_logger().info(f"Loading tomogram from: {tomogram_path}")
        try:
            with open(tomogram_path, 'rb') as f:
                data_dict = pickle.load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load tomogram pickle: {e}")
            raise

        pickled_data = pickle.dumps(data_dict)
        msg = ByteMultiArray()
        # 与 tomography.publish_tomogram_data 一致：list of bytes objects
        msg.data = [bytes([b]) for b in pickled_data]

        self.get_logger().info("Initializing planner with loaded tomogram data...")
        self.planner.update_tomogram_from_msg(msg)
        self.tomogram_received = True
        self.get_logger().info("Planner initialized from file.")

        # 起点位姿：初始为 (0, 0, 0)
        # door
        # self.start_pos = np.array([-1.0747, 6.56034, 0], dtype=np.float32)

        # face button
        # self.goal_pos = np.array([-1.00449, -1.05048, 0], dtype=np.float32)
        # # back to button
        self.start_pos = np.array([-0.828478, -2.326155, 0], dtype=np.float32)

        # # in cabin
        self.goal_pos = np.array([1.75459, -2.32253, 0], dtype=np.float32)
        self.path_pub = self.create_publisher(Path, '/pct_path2', 10)

        # 当前轨迹缓存，用于定频率发布
        self.current_traj = None
        # 1 Hz 定时器，按固定频率发布路径
        self.publish_timer = self.create_timer(1.0, self.publish_loop)

        # 订阅 /local_pose (geometry_msgs/PoseStamped)，随时更新 start_pos 并重新规划
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/local_pose',
            self.odom_callback,
            10,
        )
        self.goal = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10,
        )

        self.get_logger().info('Initial start pose is (0, 0, 0).')
        self.get_logger().info('Goal pose is fixed at (1, 6, 0).')

        # 初始化完成后，立刻尝试用原点到目标做一次规划
        self.try_plan()
    def goal_callback(self, msg: PoseStamped):
        """接收 /local_pose (PoseStamped)，更新当前起点，并重新规划。"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.goal_pos = np.array([x, y, z], dtype=np.float32)
        self.get_logger().info(f'Received odom pose as new start: {self.start_pos}')


    def odom_callback(self, msg: PoseStamped):
        """接收 /local_pose (PoseStamped)，更新当前起点，并重新规划。"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.start_pos = np.array([x, y, z], dtype=np.float32)
        self.get_logger().info(f'Received odom pose as new start: {self.start_pos}')

        # 每次收到新里程计位姿，都重新规划一次
        self.try_plan()

    def try_plan(self):
        """在 tomogram 准备好时，从当前 start_pos 规划到 goal_pos。"""
        if not self.tomogram_received:
            self.get_logger().info('Tomogram not ready yet. Cannot plan path.')
            return

        start_pos_np = self.start_pos
        end_pos_np = self.goal_pos

        
        # 如果当前起点与目标距离小于 2m，则不再重新规划，只保留/发布当前轨迹
        dist = np.linalg.norm(start_pos_np[:2] - end_pos_np[:2])
        # if dist < 2.0:
        #     self.get_logger().info(
        #         f'Distance to goal is {dist:.2f} m (< 2 m). Skip replanning, only publishing current trajectory.'
        #     )
        #     return
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
