import os
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Point

from utils import *
from sparse_planner_wrapper import SparseTomogramPlanner

sys.path.append('../')
from config import Config


class PCTPlannerDirectNode(Node):
    """Planner node that loads the sparse tomogram directly from a pickle file.

    It mimics the behavior of `plan.py` but reads
    `rsc/tomogram/scene_map_sparse.pickle` at startup and uses
    `SparseTomogramPlanner` / Sparse A* instead of the dense planner.
    """

    def __init__(self):
        super().__init__('pct_planner_direct')

        self.cfg = Config()
        self.planner = SparseTomogramPlanner(self.cfg)
        self.tomogram_received = False
        self.start_pos = None
        self.end_pos = None

        self.path_pub = self.create_publisher(Path, "/pct_path2", 10)

        # Subscribe to start and end positions
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

        # Load sparse tomogram directly
        self.get_logger().info("Loading sparse tomogram from scene_map_sparse.pickle...")
        self.planner.loadTomogram('scene_map_sparse')
        self.tomogram_received = True
        self.get_logger().info("Sparse planner initialized from file.")

        self.get_logger().info("Waiting for start position on /start_pos topic...")
        self.get_logger().info("Waiting for end position on /end_pos topic...")

        # Current trajectory, republished periodically
        self.current_traj = None
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
        """Republish the current trajectory periodically."""
        if self.current_traj is None:
            return

        path_msg = traj2ros(self.current_traj)
        self.path_pub.publish(path_msg)
        self.get_logger().debug("Republishing trajectory to /pct_path2")

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

        self.get_logger().info("All data ready. Planning sparse A* path...")

        start_pos_np = self.start_pos
        end_pos_np = self.end_pos

        traj_3d = self.planner.plan(start_pos_np[:2], end_pos_np[:2], start_pos_np[2], end_pos_np[2])
        if traj_3d is not None:
            self.current_traj = traj_3d
            self.get_logger().info("Trajectory planned successfully. Now continuously publishing to /pct_path2.")
        else:
            self.get_logger().warn("Failed to generate a trajectory.")


def main(args=None):
    rclpy.init(args=args)

    node = PCTPlannerDirectNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
