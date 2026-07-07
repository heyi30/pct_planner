import os
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped

from utils import *
from sparse_planner_wrapper import SparseTomogramPlanner

sys.path.append('../')
from config import Config


class PCTPlannerSystemNode(Node):
    def __init__(self):
        super().__init__('pct_planner_system')

        self.cfg = Config()
        self.planner = SparseTomogramPlanner(self.cfg)

        # Load sparse tomogram directly
        self.get_logger().info("Loading sparse tomogram from scene_map_sparse.pickle...")
        self.planner.loadTomogram('scene_map_sparse')
        self.tomogram_received = True
        self.get_logger().info("Sparse planner initialized from file.")

        stair_end = np.array([-28.221, -38.4062, 0.6972022652626038, 1.0], dtype=np.float32)
        lobby = np.array([-33.525770568847656, -25.410804977416992, 0.6448325514793396, 1.0], dtype=np.float32)

        self.M_loc2pct = np.array([[ 0.962,  0.273,  0.002,  11.728],
                                [-0.273,  0.962, -0.004,  -1.289],
                                [-0.003,  0.004,  1.000,   0.111],
                                [ 0.000,  0.000,  0.000,   1.000]])

        self.M_pct2loc = np.array([[ 0.962, -0.273, -0.003, -11.634],
                            [ 0.273,  0.962,  0.004,  -1.960],
                            [ 0.002, -0.004,  1.000,  -0.142],
                            [ 0.000,  0.000,  0.000,   1.000]])

        transformed_stair_end = self.M_loc2pct @ stair_end
        transformed_lobby = self.M_loc2pct @ lobby

        self.goal_pos = transformed_lobby[:3]
        self.start_pos = transformed_stair_end[:3]

        self.path_pub = self.create_publisher(Path, '/pct_path2', 10)

        # Current trajectory cache, published at fixed frequency
        self.current_traj = None
        self.publish_timer = self.create_timer(1.0, self.publish_loop)

        # Subscribe to local pose and replan on each update
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/local_pose',
            self.odom_callback,
            10,
        )

        self.get_logger().info('Initial start pose is (0, 0, 0).')
        self.get_logger().info('Goal pose is fixed.')

        # Try an initial plan from the default start
        self.try_plan()

    def odom_callback(self, msg: PoseStamped):
        """Receive /local_pose, update current start, and replan."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.start_pos = np.array([x, y, z], dtype=np.float32)
        self.get_logger().info(f'Received odometry pose as new start: {self.start_pos}')
        self.try_plan()

    def try_plan(self):
        """Plan from current start_pos to goal_pos using the sparse planner."""
        if not self.tomogram_received:
            self.get_logger().info('Tomogram not ready yet. Cannot plan path.')
            return

        start_pos_np = self.start_pos
        end_pos_np = self.goal_pos

        # If close to the goal, stop replanning and keep publishing the current trajectory
        dist = np.linalg.norm(start_pos_np[:2] - end_pos_np[:2])
        if dist < 2.0:
            self.get_logger().info(
                f'Distance to goal is {dist:.2f} m (< 2 m). Skip replanning, only publishing current trajectory.'
            )
            return

        self.get_logger().info(
            f'Planning sparse A* path from start {start_pos_np.tolist()} to goal {end_pos_np.tolist()}...'
        )

        traj_3d = self.planner.plan(
            start_pos_np[:2],
            end_pos_np[:2],
            start_pos_np[2],
            end_pos_np[2],
        )

        if traj_3d is not None:
            self.current_traj = []
            for p in traj_3d:
                p_homo = np.array([p[0], p[1], p[2], 1.0])
                p_loc = self.M_pct2loc @ p_homo
                p_new = p_loc[:3]
                print(p_new)
                self.current_traj.append(p_new)
            self.get_logger().info('Sparse A* path planned. Publishing at 1 Hz to /pct_path2.')
        else:
            self.get_logger().warn('Failed to generate a trajectory.')

    def publish_loop(self):
        """Publish the current trajectory at a fixed frequency."""
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
