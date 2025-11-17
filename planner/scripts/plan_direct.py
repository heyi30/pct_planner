import os
import sys
import pickle
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import ByteMultiArray
from geometry_msgs.msg import Point

from utils import *
from planner_wrapper import TomogramPlanner

sys.path.append('../')
from config import Config


class PCTPlannerDirectNode(Node):
    """Planner node that loads tomogram data directly from a pickle file.

    It mimics the behavior of `plan.py` but instead of subscribing to
    `/tomogram_data`, it reads `/home/hanjiatong/PctPlanner/rsc/tomogram/scene_map.pickle`
    once at startup and feeds it to TomogramPlanner via the same
    `update_tomogram_from_msg` interface by constructing a ByteMultiArray
    message with identical layout to `tomography.publish_tomogram_data`.
    """

    def __init__(self, tomogram_path: str):
        super().__init__('pct_planner_direct')

        self.cfg = Config()
        self.planner = TomogramPlanner(self.cfg)
        self.tomogram_received = False
        self.start_pos = None
        self.end_pos = None

        self.path_pub = self.create_publisher(Path, "/pct_path2", 10)

        # Subscribe to start and end positions as before
        self.start_pos_sub = self.create_subscription(
            Point,
            "/start_pos",
            self.start_pos_callback,
            10)
        self.end_pos_sub = self.create_subscription(
            Point,
            "/end_pos",
            self.end_pos_callback,
            10)

        # Load tomogram pickle and construct a ByteMultiArray like tomography.publish_tomogram_data
        self.get_logger().info(f"Loading tomogram from: {tomogram_path}")
        try:
            with open(tomogram_path, 'rb') as f:
                data_dict = pickle.load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load tomogram pickle: {e}")
            raise

        pickled_data = pickle.dumps(data_dict)
        msg = ByteMultiArray()
        # Follow tomography.publish_tomogram_data: list of bytes objects
        msg.data = [bytes([b]) for b in pickled_data]

        # Initialize planner with this tomogram once
        self.get_logger().info("Initializing planner with loaded tomogram data...")
        self.planner.update_tomogram_from_msg(msg)
        self.tomogram_received = True
        self.get_logger().info("Planner initialized from file.")

        self.get_logger().info("Waiting for start position on /start_pos topic...")
        self.get_logger().info("Waiting for end position on /end_pos topic...")

        # 当前规划出的轨迹，用于循环发布
        self.current_traj = None
        # 定时器，周期性地将当前轨迹发布到 /pct_path
        self.publish_timer = self.create_timer(0.5, self.publish_loop)

    def start_pos_callback(self, msg: Point):
        self.start_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.get_logger().info(f"Received start position: {self.start_pos}")
        self.try_plan()

    def end_pos_callback(self, msg: Point):
        self.end_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.get_logger().info(f"Received end position: {self.end_pos}")
        self.try_plan()

    def publish_loop(self):
        """周期性将当前轨迹发布到 /pct_path。"""
        if self.current_traj is None:
            return

        path_msg = traj2ros(self.current_traj)
        self.path_pub.publish(path_msg)
        # 用 debug 级别避免刷屏
        self.get_logger().debug("Republishing trajectory to /pct_path")

    def try_plan(self):
        """Attempt planning when tomogram, start, and end are all available."""
        if not self.tomogram_received:
            self.get_logger().info("Tomogram not ready yet.")
            return

        if self.start_pos is None:
            self.get_logger().info("Waiting for start position...")
            return

        if self.end_pos is None:
            self.get_logger().info("Waiting for end position...")
            return

        self.get_logger().info("All data ready. Planning trajectory...")

        start_pos_np = self.start_pos
        end_pos_np = self.end_pos

        traj_3d = self.planner.plan(start_pos_np[:2], end_pos_np[:2], start_pos_np[2] + 0.5, end_pos_np[2] + 0.5)
        if traj_3d is not None:
            # 保存轨迹，由定时器循环发布
            self.current_traj = traj_3d
            # 如果需要保存 pcd，可以解除下面注释
            # save_traj_as_pcd(traj_3d, 'trajectory.pcd')
            self.get_logger().info("Trajectory planned successfully. Now continuously publishing to /pct_path.")

            # 如需每次新起终点都重新规划，可以选择是否重置 start/end
            # 这里不重置，允许多次接收新的起终点覆盖 current_traj
        else:
            self.get_logger().warn("Failed to generate a trajectory.")


def main(args=None):
    rclpy.init(args=args)

    # Default tomogram file path as requested
    tomogram_path = "/home/hanjiatong/PctPlanner/rsc/tomogram/scene_map.pickle"

    node = PCTPlannerDirectNode(tomogram_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
