#!/usr/bin/env python3
"""Publish JSON waypoints as a visualization_msgs/MarkerArray in RViz.

Reads a JSON list of waypoint entries (each with "name", "pose" [x, y, z]
and optional "radius") and publishes them as markers: a sphere per point,
a floating text label with the name, a semi-transparent disk for the
radius, and a line strip connecting the points in order.

Usage:
    source /opt/ros/humble/setup.bash
    python3 pub_json_markers.py /path/to/waypoints.json
"""

import argparse
import json


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


def create_marker_publisher_class():
    from geometry_msgs.msg import Point
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from visualization_msgs.msg import Marker, MarkerArray

    def make_marker(ns, marker_id, marker_type, frame_id, stamp):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def set_color(marker, r, g, b, a):
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a

    class JsonWaypointMarkerPublisher(Node):
        def __init__(self, entries, topic, frame_id, rate):
            super().__init__('json_waypoint_marker_publisher')
            self.entries = entries
            self.frame_id = frame_id

            qos_profile = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.publisher = self.create_publisher(MarkerArray, topic, qos_profile)
            self.timer = self.create_timer(1.0 / rate, self.publish_markers)
            self.publish_markers()
            self.get_logger().info(
                f"Publishing markers for {len(entries)} waypoints to {topic} at {rate} Hz"
            )

        def build_markers(self):
            stamp = self.get_clock().now().to_msg()
            markers = []

            # Order line through all points.
            line = make_marker('waypoint_order', 0, Marker.LINE_STRIP,
                               self.frame_id, stamp)
            line.scale.x = 0.05
            set_color(line, 0.1, 0.9, 0.1, 0.9)
            for entry in self.entries:
                point = Point()
                point.x, point.y, point.z = (float(v) for v in entry['pose'])
                line.points.append(point)
            markers.append(line)

            for i, entry in enumerate(self.entries):
                x, y, z = (float(v) for v in entry['pose'])
                name = str(entry.get('name', f'wp_{i}'))
                radius = float(entry.get('radius', 0.0))

                sphere = make_marker('waypoints', i, Marker.SPHERE,
                                     self.frame_id, stamp)
                sphere.pose.position.x = x
                sphere.pose.position.y = y
                sphere.pose.position.z = z
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
                set_color(sphere, 1.0, 0.4, 0.0, 1.0)
                markers.append(sphere)

                if radius > 0.0:
                    disk = make_marker('waypoint_radius', i, Marker.CYLINDER,
                                       self.frame_id, stamp)
                    disk.pose.position.x = x
                    disk.pose.position.y = y
                    disk.pose.position.z = z
                    disk.scale.x = disk.scale.y = 2.0 * radius
                    disk.scale.z = 0.02
                    set_color(disk, 0.2, 0.5, 1.0, 0.25)
                    markers.append(disk)

                label = make_marker('waypoint_names', i, Marker.TEXT_VIEW_FACING,
                                    self.frame_id, stamp)
                label.pose.position.x = x
                label.pose.position.y = y
                label.pose.position.z = z + 0.8
                label.scale.z = 1.2
                set_color(label, 1.0, 1.0, 1.0, 1.0)
                label.text = name
                markers.append(label)

            return markers

        def publish_markers(self):
            array_msg = MarkerArray()
            array_msg.markers = self.build_markers()
            self.publisher.publish(array_msg)

    return JsonWaypointMarkerPublisher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_file', type=str,
                        help="input JSON file with waypoint entries")
    parser.add_argument('--topic', type=str, default='/pct_waypoint_markers',
                        help="ROS 2 topic for MarkerArray. Default: /pct_waypoint_markers")
    parser.add_argument('--frame-id', type=str, default='map',
                        help="frame ID for the markers. Default: map")
    parser.add_argument('--rate', type=float, default=1.0,
                        help="publish rate in Hz. Default: 1.0")
    return parser.parse_args()


def main():
    args = parse_args()
    entries = load_waypoints(args.json_file)
    print(f"Loaded {len(entries)} waypoints from {args.json_file}")

    import rclpy
    rclpy.init()
    node = None
    try:
        publisher_class = create_marker_publisher_class()
        node = publisher_class(
            entries=entries,
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
