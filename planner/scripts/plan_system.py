import sys
import os
import argparse
import pickle
import numpy as np

# 自动设置 LD_LIBRARY_PATH，确保 GTSAM 库可以被加载
# 必须在任何其他 import 之前设置，并重新执行脚本
_gtsam_lib_path = os.path.expanduser("~/numa/PctPlanner/planner/lib/3rdparty/gtsam-4.1.1/install/lib")
_current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

if _gtsam_lib_path not in _current_ld_path:
    os.environ["LD_LIBRARY_PATH"] = _current_ld_path + ":" + _gtsam_lib_path if _current_ld_path else _gtsam_lib_path
    # 重新执行脚本以使 LD_LIBRARY_PATH 生效
    os.execv(sys.executable, [sys.executable] + sys.argv)

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Point
from pct_planner_msgs.srv import PlanPath as PathSrv

from utils import *
from planner_wrapper import TomogramPlanner

sys.path.append('../')
from config import Config

import time
import pandas as pd

# 本地 tomogram 数据文件路径
TOMOGRAM_FILE_PATH = "/home/nuc/numa/PctPlanner/rsc/tomogram/nyby_underground.pickle"

def save_traj_as_csv(traj, filename=None):
    """
    Saves a trajectory to a .csv file with a timestamped filename.
    """
    if filename is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"trajectory_{timestamp}.csv"
    
    df = pd.DataFrame(traj, columns=['x', 'y', 'z'])
    df.to_csv(filename, index=False)
    return filename

class PCTPlannerNode(Node):
    def __init__(self):
        super().__init__('pct_planner')

        self.cfg = Config()
        self.planner = TomogramPlanner(self.cfg)
        self.tomogram_received = False
        self.start_pos = None
        self.end_pos = None
        self.server_ = self.create_service(
            PathSrv,
            '/plan_path',
            self.handle_plan_path)
        # 从本地文件加载 tomogram 数据
        self._load_tomogram_from_file(TOMOGRAM_FILE_PATH)

        self.path_pub = self.create_publisher(Path, "/global_trajectory", 1)
        self.waypoints_sub = self.create_subscription(
            Path,
            "/waypoints",
            self.waypoints_callback,
            1)
        self.start_pos_sub = self.create_subscription(
            Point,
            "/start_pos",
            self.start_pos_callback,
            1)
        self.end_pos_sub = self.create_subscription(
            Point,
            "/end_pos",
            self.end_pos_callback,
            1)
        self.get_logger().info("Tomogram loaded from local file.")
        self.get_logger().info("Waiting for waypoints on /waypoints topic...")
        self.get_logger().info("Waiting for start position on /start_pos topic...")
        self.get_logger().info("Waiting for end position on /end_pos topic...")

    def _load_tomogram_from_file(self, file_path):
        """从本地 pickle 文件加载 tomogram 数据"""
        if not os.path.exists(file_path):
            self.get_logger().error(f"Tomogram file not found: {file_path}")
            raise FileNotFoundError(f"Tomogram file not found: {file_path}")
        
        self.get_logger().info(f"Loading tomogram from: {file_path}")
        with open(file_path, 'rb') as handle:
            data_dict = pickle.load(handle)
            self.planner._initialize_from_dict(data_dict)
        self.tomogram_received = True
        self.get_logger().info("Tomogram data loaded successfully.")

    def plan_multi_points(self, waypoints):
        """
        Plans a path through multiple waypoints.
        waypoints: List of np.array([x, y, z])
        """
        if not self.tomogram_received:
            self.get_logger().info("Waiting for tomogram data...")
            return None

        if len(waypoints) < 2:
            self.get_logger().warn("Need at least 2 points for planning.")
            return None

        full_traj = []
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i+1]
            
            self.get_logger().info(f"Planning segment {i}: from {p1} to {p2}")
            
            try:
                # Use the same logic as before: z + 0.5
                traj = self.planner.plan(p1[:2], p2[:2], p1[2] + 0.5, p2[2] + 0.5)
            except Exception as e:
                self.get_logger().error(f"Error during planning segment {i}: {e}")
                return None
            
            if traj is not None:
                if len(full_traj) > 0:
                    # Avoid duplicating the connection point
                    full_traj.extend(traj[1:])
                else:
                    full_traj.extend(traj)
            else:
                self.get_logger().error(f"Failed to plan segment {i} from {p1} to {p2}. Multi-point planning aborted.")
                return None

        return np.array(full_traj)

    def waypoints_callback(self, msg):
        self.get_logger().info(f"Received {len(msg.poses)} waypoints.")
        waypoints = []
        for pose_stamped in msg.poses:
            p = pose_stamped.pose.position
            waypoints.append(np.array([p.x, p.y, p.z], dtype=np.float32))
        
        traj_3d = self.plan_multi_points(waypoints)
        if traj_3d is not None:
            path_msg = traj2ros(traj_3d)
            path_msg.header.frame_id = "map"
            self.path_pub.publish(path_msg)
            csv_file = save_traj_as_csv(traj_3d)
            self.get_logger().info(f"Multi-point trajectory published and saved to {csv_file}.")
    def handle_plan_path(self, request, response):
        self.get_logger().info(f"Received {len(request.path.poses)} waypoints via service.")
        waypoints = []
        for pose_stamped in request.path.poses:
            p = pose_stamped.pose.position
            waypoints.append(np.array([p.x, p.y, p.z], dtype=np.float32))
        
        traj_3d = self.plan_multi_points(waypoints)
        if traj_3d is not None:
            path_msg = traj2ros(traj_3d)
            path_msg.header.frame_id = "map"
            self.path_pub.publish(path_msg)
            # csv_file = save_traj_as_csv(traj_3d)
            # self.get_logger().info(f"Trajectory published and saved to {csv_file}.")
            response.path = path_msg
            response.success = True
            self.get_logger().info("Multi-point trajectory generated and published.")
        else:
            response.success = False
            self.get_logger().error("Failed to generate multi-point trajectory.")
        return response

    def start_pos_callback(self, msg):
        self.start_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.get_logger().info(f"Received start position: {self.start_pos}")

    def end_pos_callback(self, msg):
        self.end_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.get_logger().info(f"Received end position: {self.end_pos}")
        
        if self.start_pos is None:
            self.get_logger().info("Waiting for start position...")
            return
        
        traj_3d = self.plan_multi_points([self.start_pos, self.end_pos])
        
        if traj_3d is not None:
            path_msg = traj2ros(traj_3d)
            path_msg.header.frame_id = "map"
            self.path_pub.publish(path_msg)
            csv_file = save_traj_as_csv(traj_3d)
            self.get_logger().info(f"Trajectory published and saved to {csv_file}.")
            # Reset for next planning cycle
            self.start_pos = None
            self.end_pos = None
        else:
            self.get_logger().warn("Failed to generate a trajectory.")

def main(args=None):
    rclpy.init(args=args)
    node = PCTPlannerNode()
    # We don't need to spin the node if it's just publishing once.
    # rclpy.spin(node) 
    # node.destroy_node()
    # rclpy.shutdown()
    # But if we want to keep it alive to check the published path, we can spin it.
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
