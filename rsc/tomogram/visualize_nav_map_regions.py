#!/usr/bin/env python3
"""Visualize nav_map height-layer regions from a JSON file in RViz2.

Reads the output of split_nav_map_by_height.py and publishes the bboxes as a
MarkerArray. Each region is drawn as a wireframe box; region id, height range
and node count can optionally be shown as text labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


RSG_ROOT = Path(__file__).resolve().parents[2]


def generate_color(seed: int) -> tuple[float, float, float]:
    """Generate a deterministic bright color from an integer seed."""
    rng = random.Random(seed)
    h = rng.random()
    s = 0.8 + rng.random() * 0.2
    v = 0.8 + rng.random() * 0.2
    c = v * s
    x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
    m = v - c
    if h < 1.0 / 6.0:
        r, g, b = c, x, 0.0
    elif h < 2.0 / 6.0:
        r, g, b = x, c, 0.0
    elif h < 3.0 / 6.0:
        r, g, b = 0.0, c, x
    elif h < 4.0 / 6.0:
        r, g, b = 0.0, x, c
    elif h < 5.0 / 6.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return r + m, g + m, b + m


def resolve_json(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    if p.exists():
        return p.resolve()
    return (RSG_ROOT / "rsc" / "tomogram" / p).resolve()


def load_regions(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "regions" not in data:
        raise ValueError("expected JSON object with 'regions' key")
    return data["regions"]


def make_wireframe_marker(
    region: dict,
    marker_id: int,
    frame_id: str,
    stamp,
    rgb: tuple[float, float, float] | None = None,
    line_width: float = 0.05,
) -> Marker:
    """Return a LINE_LIST marker outlining the region bbox."""
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = "nav_map_regions_bbox"
    m.id = marker_id
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = line_width
    if rgb is None:
        rgb = generate_color(region["id"])
    m.color.r, m.color.g, m.color.b = rgb
    m.color.a = 1.0

    b = region["bbox"]
    corners = [
        Point(x=b["min_x"], y=b["min_y"], z=b["min_z"]),
        Point(x=b["max_x"], y=b["min_y"], z=b["min_z"]),
        Point(x=b["max_x"], y=b["max_y"], z=b["min_z"]),
        Point(x=b["min_x"], y=b["max_y"], z=b["min_z"]),
        Point(x=b["min_x"], y=b["min_y"], z=b["max_z"]),
        Point(x=b["max_x"], y=b["min_y"], z=b["max_z"]),
        Point(x=b["max_x"], y=b["max_y"], z=b["max_z"]),
        Point(x=b["min_x"], y=b["max_y"], z=b["max_z"]),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        m.points.append(corners[i])
        m.points.append(corners[j])
    return m


def make_label_marker(
    region: dict,
    marker_id: int,
    frame_id: str,
    stamp,
    rgb: tuple[float, float, float] | None = None,
    text_height: float = 0.4,
) -> Marker:
    """Return a TEXT_VIEW_FACING marker with region metadata."""
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = "nav_map_regions_label"
    m.id = marker_id
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.scale.z = text_height
    if rgb is None:
        rgb = generate_color(region["id"])
    m.color.r, m.color.g, m.color.b = rgb
    m.color.a = 1.0

    c = region["center"]
    m.pose.position.x = float(c["x"])
    m.pose.position.y = float(c["y"])
    m.pose.position.z = float(c["z"]) + text_height * 0.5
    m.pose.orientation.w = 1.0

    b = region["bbox"]
    m.text = (
        f"id:{region['id']} comp:{region['component_id']}\n"
        f"nodes:{region['node_count']}\n"
        f"z:[{b['min_z']:.2f}, {b['max_z']:.2f}]"
    )
    return m


def make_center_marker(
    region: dict,
    marker_id: int,
    frame_id: str,
    stamp,
    rgb: tuple[float, float, float] | None = None,
    diameter: float = 0.15,
) -> Marker:
    """Return a SPHERE marker at the region center."""
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = "nav_map_regions_center"
    m.id = marker_id
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.scale.x = diameter
    m.scale.y = diameter
    m.scale.z = diameter
    if rgb is None:
        rgb = generate_color(region["id"])
    m.color.r, m.color.g, m.color.b = rgb
    m.color.a = 1.0

    c = region["center"]
    m.pose.position.x = float(c["x"])
    m.pose.position.y = float(c["y"])
    m.pose.position.z = float(c["z"])
    m.pose.orientation.w = 1.0
    return m


def make_delete_all_marker(frame_id: str, stamp) -> Marker:
    """Return a marker that clears all markers in both namespaces."""
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.action = Marker.DELETEALL
    return m


class RegionVisualizer(Node):
    def __init__(
        self,
        regions: list[dict],
        frame_id: str,
        topic: str,
        rate: float,
        with_labels: bool,
        with_centers: bool,
        line_width: float,
        label_height: float,
        center_diameter: float,
    ):
        super().__init__("nav_map_region_visualizer")
        self.regions = regions
        self.frame_id = frame_id
        self.with_labels = with_labels
        self.with_centers = with_centers
        self.line_width = line_width
        self.label_height = label_height
        self.center_diameter = center_diameter

        self.pub = self.create_publisher(MarkerArray, topic, 10)
        self.timer = self.create_timer(1.0 / rate, self.publish)
        self.get_logger().info(
            f"visualizing {len(regions)} regions on '{topic}' at {rate} Hz "
            f"(frame_id='{frame_id}', labels={with_labels}, centers={with_centers})"
        )

    def publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        markers: list[Marker] = []

        # Publish a DELETEALL first so stale markers are removed if the region
        # count changes between runs.
        markers.append(make_delete_all_marker(self.frame_id, stamp))

        for region in self.regions:
            rid = region["id"]
            rgb = generate_color(rid)
            markers.append(
                make_wireframe_marker(
                    region, rid, self.frame_id, stamp, rgb=rgb, line_width=self.line_width
                )
            )
            if self.with_centers:
                markers.append(
                    make_center_marker(
                        region, rid, self.frame_id, stamp, rgb=rgb, diameter=self.center_diameter
                    )
                )
            if self.with_labels:
                markers.append(
                    make_label_marker(
                        region, rid, self.frame_id, stamp, rgb=rgb, text_height=self.label_height
                    )
                )

        self.pub.publish(MarkerArray(markers=markers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize nav_map height-layer regions from a JSON file in RViz2."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="scene_map_sparse_planner_height_regions.json",
        help="input JSON file (absolute path or basename under rsc/tomogram/)",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default="map",
        help="RViz2 frame_id for the markers",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="nav_map_region_markers",
        help="MarkerArray topic name",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="publication rate in Hz",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="also publish id/height/node-count text labels",
    )
    parser.add_argument(
        "--with-centers",
        action="store_true",
        help="also publish sphere markers at region centers",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=0.05,
        help="bbox wireframe line width",
    )
    parser.add_argument(
        "--label-height",
        type=float,
        default=0.4,
        help="text label height",
    )
    parser.add_argument(
        "--center-diameter",
        type=float,
        default=0.15,
        help="center sphere diameter",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = resolve_json(args.json)
    regions = load_regions(json_path)
    if not regions:
        print("No regions to visualize.", file=sys.stderr)
        return 1

    rclpy.init(args=None)
    node = RegionVisualizer(
        regions=regions,
        frame_id=args.frame_id,
        topic=args.topic,
        rate=args.rate,
        with_labels=args.with_labels,
        with_centers=args.with_centers,
        line_width=args.line_width,
        label_height=args.label_height,
        center_diameter=args.center_diameter,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
