#!/usr/bin/env python3
"""Plan a trajectory through JSON waypoints and publish it to ROS 2.

Reads a JSON list of waypoint entries (each with a "pose" [x, y, z]),
plans between every two consecutive poses in order, concatenates the
segments into one trajectory, then publishes it as nav_msgs/Path at a
fixed rate (default 1 Hz).

Usage:
    source /opt/ros/humble/setup.bash
    python3 pub_json_path.py /path/to/waypoints.json --map scene_map_sparse
"""

import argparse
import json

import numpy as np

from Planner import Planner


def load_waypoints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of waypoint entries")
    for i, entry in enumerate(data):
        pose = entry.get('pose')
        if not (isinstance(pose, list) and len(pose) == 3):
            raise ValueError(
                f"entry {i} ({entry.get('name')!r}) has no valid 'pose' [x, y, z]"
            )
    return data


def plan_trajectory(planner, entries):
    """Plan between consecutive poses and concatenate into one Nx3 array."""
    segments = []
    n_segments = len(entries) - 1
    for i in range(n_segments):
        start = entries[i]['pose']
        goal = entries[i + 1]['pose']
        print(f"\n[{i + 1}/{n_segments}] {entries[i].get('name')} -> {entries[i + 1].get('name')}")
        print(f"    start = {start}, goal = {goal}")
        path = planner.plan(start, goal)
        if path is None:
            raise RuntimeError(
                f"planning failed between {entries[i].get('name')!r} "
                f"and {entries[i + 1].get('name')!r}"
            )
        print(f"    segment waypoints: {path.shape[0]}")
        # Drop the joint point duplicated by the previous segment's last point.
        segments.append(path[1:] if segments else path)
    return np.concatenate(segments, axis=0)


def create_path_publisher_class():
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path as RosPath
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )

    class JsonPathPublisher(Node):
        def __init__(self, trajectory, topic, frame_id, rate):
            super().__init__('json_path_publisher')
            self.trajectory = trajectory
            self.frame_id = frame_id

            qos_profile = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.publisher = self.create_publisher(RosPath, topic, qos_profile)
            self.timer = self.create_timer(1.0 / rate, self.publish_path)
            self.publish_path()
            self.get_logger().info(
                f"Publishing {trajectory.shape[0]} waypoints to {topic} at {rate} Hz"
            )

        def publish_path(self):
            stamp = self.get_clock().now().to_msg()
            path_msg = RosPath()
            path_msg.header.frame_id = self.frame_id
            path_msg.header.stamp = stamp
            for x, y, z in self.trajectory:
                pose = PoseStamped()
                pose.header.frame_id = self.frame_id
                pose.header.stamp = stamp
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = float(z)
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.publisher.publish(path_msg)

    return JsonPathPublisher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_file', type=str,
                        help="input JSON file with waypoint entries")
    parser.add_argument('--map', type=str, default='scene_map_sparse',
                        help="sparse tomogram pickle name under rsc/tomogram/ (without .pickle)")
    parser.add_argument('--topic', type=str, default='/pct_path',
                        help="ROS 2 topic to publish nav_msgs/Path on. Default: /pct_path")
    parser.add_argument('--frame-id', type=str, default='map',
                        help="frame ID for the Path header and poses. Default: map")
    parser.add_argument('--rate', type=float, default=1.0,
                        help="publish rate in Hz. Default: 1.0")
    return parser.parse_args()


def main():
    args = parse_args()

    entries = load_waypoints(args.json_file)
    if len(entries) < 2:
        raise ValueError("need at least two waypoints to plan between")

    print(f"Loaded {len(entries)} waypoints from {args.json_file}")
    print(f"Loading map: {args.map}.pickle")
    planner = Planner(map_name=args.map)

    trajectory = plan_trajectory(planner, entries)
    print(f"\nTotal trajectory waypoints: {trajectory.shape[0]}")

    import rclpy
    rclpy.init()
    node = None
    try:
        publisher_class = create_path_publisher_class()
        node = publisher_class(
            trajectory=trajectory,
            topic=args.topic,
            frame_id=args.frame_id,
            rate=args.rate,
        )
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
