#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path as FilePath


DEFAULT_CSV_PATH = FilePath(__file__).with_name("path.csv")


def load_path_points(csv_path: FilePath) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []

    with csv_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_fields = {"x", "y", "z"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain x,y,z columns")

        for row_number, row in enumerate(reader, start=2):
            try:
                points.append((float(row["x"]), float(row["y"]), float(row["z"])))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value in {csv_path}:{row_number}") from exc

    if not points:
        raise ValueError(f"{csv_path} does not contain any path points")

    return points


def create_csv_path_publisher_class():
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path as RosPath
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    class CsvPathPublisher(Node):
        def __init__(
            self,
            csv_path: FilePath,
            topic: str,
            frame_id: str,
            publish_rate: float,
        ) -> None:
            super().__init__("csv_path_publisher")

            self.csv_path = csv_path
            self.frame_id = frame_id
            self.points = load_path_points(csv_path)

            qos_profile = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.publisher = self.create_publisher(RosPath, topic, qos_profile)

            self.publish_path()
            if publish_rate > 0.0:
                self.timer = self.create_timer(1.0 / publish_rate, self.publish_path)
            else:
                self.timer = None

            self.get_logger().info(
                f"Loaded {len(self.points)} points from {csv_path} and publishing to {topic}"
            )

        def build_path_message(self):
            stamp = self.get_clock().now().to_msg()

            path_msg = RosPath()
            path_msg.header.frame_id = self.frame_id
            path_msg.header.stamp = stamp

            for x, y, z in self.points:
                pose = PoseStamped()
                pose.header.frame_id = self.frame_id
                pose.header.stamp = stamp
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)

            return path_msg

        def publish_path(self) -> None:
            self.publisher.publish(self.build_path_message())

    return CsvPathPublisher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish planner/scripts/path.csv as a ROS 2 nav_msgs/Path."
    )
    parser.add_argument(
        "--csv",
        type=FilePath,
        default=DEFAULT_CSV_PATH,
        help=f"CSV file containing x,y,z columns. Default: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--topic",
        default="/pct_path",
        help="ROS 2 topic to publish nav_msgs/Path on. Default: /pct_path",
    )
    parser.add_argument(
        "--frame-id",
        default="map",
        help="Frame ID for the Path header and poses. Default: map",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Publish rate in Hz. Use 0 to publish once and keep the node alive. Default: 1.0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import rclpy

    rclpy.init()
    node = None
    try:
        csv_path_publisher_class = create_csv_path_publisher_class()
        node = csv_path_publisher_class(
            csv_path=args.csv.expanduser().resolve(),
            topic=args.topic,
            frame_id=args.frame_id,
            publish_rate=args.rate,
        )
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
