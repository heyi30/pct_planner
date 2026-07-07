import sys
import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Point

from utils import *
from sparse_planner_wrapper import SparseTomogramPlanner

sys.path.append('../')
from config import Config

def save_traj_as_pcd(traj, filename):
    """
    Saves a trajectory to a .pcd file.
    """
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

class PCTPlannerNode(Node):
    def __init__(self):
        super().__init__('pct_planner')

        self.cfg = Config()
        self.planner = SparseTomogramPlanner(self.cfg)
        self.planner.loadTomogram('scene_map_sparse')
        self.tomogram_received = True

        self.start_pos = None
        self.end_pos = None

        self.path_pub = self.create_publisher(Path, "/pct_path", 10)
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
        self.get_logger().info("Sparse tomogram loaded from scene_map_sparse.pickle.")
        self.get_logger().info("Waiting for start position on /start_pos topic...")
        self.get_logger().info("Waiting for end position on /end_pos topic...")

    def start_pos_callback(self, msg):
        self.start_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.get_logger().info(f"Received start position: {self.start_pos}")
        self.try_plan()

    def end_pos_callback(self, msg):
        self.end_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.get_logger().info(f"Received end position: {self.end_pos}")
        self.try_plan()

    def try_plan(self):
        if not self.tomogram_received:
            self.get_logger().info("Tomogram not ready yet.")
            return

        if self.start_pos is None:
            self.get_logger().info("Waiting for start position...")
            return

        if self.end_pos is None:
            self.get_logger().info("Waiting for end position...")
            return

        self.get_logger().info("All data received. Planning sparse A* path...")

        start_pos_np = self.start_pos
        end_pos_np = self.end_pos

        traj_3d = self.planner.plan(start_pos_np[:2], end_pos_np[:2], start_pos_np[2], end_pos_np[2])
        if traj_3d is not None:
            path_msg = traj2ros(traj_3d)
            self.path_pub.publish(path_msg)
            save_traj_as_pcd(traj_3d, 'trajectory.pcd')
            self.get_logger().info("Trajectory saved to trajectory.pcd")
            self.get_logger().info("Trajectory published to /pct_path")
            # Reset for next planning cycle
            self.start_pos = None
            self.end_pos = None
        else:
            self.get_logger().warn("Failed to generate a trajectory.")

def main(args=None):
    rclpy.init(args=args)
    node = PCTPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
