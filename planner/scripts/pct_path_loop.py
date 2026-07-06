import csv
import math
import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped


# Path to the trajectory CSV file (columns: x, y, z)
TRAJECTORY_CSV = '/home/nuc/numa/PctPlanner/planner/scripts/trajectory_20260506-145052.csv'

ODOM_TOPIC = '/odom'
PATH_TOPIC = '/global_trajectory'
FRAME_ID = 'map'

ARRIVAL_THRESHOLD = 1.0
REPUBLISH_COOLDOWN_SEC = 3.0


def load_waypoints(csv_path: str):
    """Load waypoints from a CSV file with x,y,z columns. Returns list of (x,y,z) tuples."""
    waypoints = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            waypoints.append((float(row['x']), float(row['y']), float(row['z'])))
    return waypoints


def make_pose(xyz, frame_id, stamp):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = float(xyz[0])
    pose.pose.position.y = float(xyz[1])
    pose.pose.position.z = float(xyz[2])
    pose.pose.orientation.w = 1.0
    return pose


class PctPathLoopNode(Node):
    def __init__(self):
        super().__init__('pct_path_loop')

        # Load trajectory from CSV and keep a forward copy
        self._forward_waypoints = load_waypoints(TRAJECTORY_CSV)
        if not self._forward_waypoints:
            raise RuntimeError(f'No waypoints found in {TRAJECTORY_CSV}')

        self.waypoints = list(self._forward_waypoints)  # current direction
        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints from {os.path.basename(TRAJECTORY_CSV)}'
        )

        self.path_pub = self.create_publisher(Path, PATH_TOPIC, 10)
        self.odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self.odom_cb, 10)

        self.last_publish_time = self.get_clock().now()
        self.timer = self.create_timer(1.0, self.publish_path_once)
        self._published_initial = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def publish_path_once(self):
        if self._published_initial:
            self.timer.cancel()
            return
        self.publish_path()
        self._published_initial = True
        self.timer.cancel()

    def publish_path(self):
        now = self.get_clock().now()
        stamp = now.to_msg()
        msg = Path()
        msg.header.frame_id = FRAME_ID
        msg.header.stamp = stamp
        for wp in self.waypoints:
            msg.poses.append(make_pose(wp, FRAME_ID, stamp))
        self.path_pub.publish(msg)
        self.last_publish_time = now
        start = self.waypoints[0]
        end = self.waypoints[-1]
        self.get_logger().info(
            f'Published path ({len(self.waypoints)} pts): '
            f'start=({start[0]:.2f},{start[1]:.2f}) -> '
            f'end=({end[0]:.2f},{end[1]:.2f})'
        )

    # ------------------------------------------------------------------
    # Odometry callback: detect arrival at current endpoint
    # ------------------------------------------------------------------

    def odom_cb(self, msg: Odometry):
        if not self._published_initial:
            return

        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        end = self.waypoints[-1]
        dist = math.sqrt((px - end[0])**2 + (py - end[1])**2 )

        if dist > ARRIVAL_THRESHOLD:
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_publish_time).nanoseconds * 1e-9
        if elapsed < REPUBLISH_COOLDOWN_SEC:
            return

        self.get_logger().info(
            f'Reached endpoint (dist={dist:.3f} m). Reversing trajectory and republishing.'
        )
        self.waypoints = list(reversed(self.waypoints))
        self.publish_path()


def main():
    rclpy.init()
    node = PctPathLoopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
