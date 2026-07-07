#!/usr/bin/env python3
"""Publish the sparse tomogram as a ROS 2 PointCloud2 for RViz visualization.

Example:
    cd rsc/tomogram
    python3 publish_sparse_tomogram.py

RViz:
    - Add a PointCloud2 display.
    - Set the topic to /sparse_tomogram.
    - Set the Fixed Frame to "map".
    - Color by the 'trav' or 'gateway' channel as desired.
"""

import os
import sys
import pickle
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

rsg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SPARSE_PATH = os.path.join(rsg_root, 'rsc', 'tomogram', 'scene_map_sparse.pickle')

POINT_FIELDS = [
    pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
    pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
    pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
    pc2.PointField(name='trav', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
    pc2.PointField(name='gateway', offset=16, datatype=pc2.PointField.FLOAT32, count=1),
]


class SparseTomogramPublisher(Node):
    def __init__(self, pickle_path, topic, frame_id, publish_rate):
        super().__init__('sparse_tomogram_publisher')

        self.pickle_path = pickle_path
        self.frame_id = frame_id
        self.pub = self.create_publisher(PointCloud2, topic, 1)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish)

        self.get_logger().info(f'Loading sparse tomogram from: {pickle_path}')
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        if data.get('format') != 'tomogram_sparse_v1':
            raise ValueError('expected tomogram_sparse_v1')

        resolution = float(data['resolution'])
        center = np.asarray(data['center'], dtype=np.float64)
        indices = np.asarray(data['indices'], dtype=np.int32)
        elev_g = np.asarray(data['elev_g'], dtype=np.float32)
        trav = np.asarray(data['trav'], dtype=np.float32)
        gateway = np.asarray(data['gateway'], dtype=np.int8)

        map_dim_y, map_dim_x = int(data['shape'][1]), int(data['shape'][2])
        offset_x = map_dim_x / 2.0
        offset_y = map_dim_y / 2.0

        rows = indices[:, 1]
        cols = indices[:, 2]

        xs = (cols - offset_x) * resolution + center[0]
        ys = (rows - offset_y) * resolution + center[1]
        zs = elev_g

        self.points = np.stack([xs, ys, zs, trav, gateway], axis=1).astype(np.float32)

        self.get_logger().info(
            f'Loaded {self.points.shape[0]} sparse nodes, publishing on {topic} at {publish_rate} Hz'
        )

    def publish(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        msg = pc2.create_cloud(header, POINT_FIELDS, self.points)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    import argparse
    parser = argparse.ArgumentParser(description='Publish sparse tomogram as PointCloud2')
    parser.add_argument('--pickle', type=str, default=DEFAULT_SPARSE_PATH,
                        help='Path to scene_map_sparse.pickle')
    parser.add_argument('--topic', type=str, default='/sparse_tomogram',
                        help='ROS topic to publish on')
    parser.add_argument('--frame', type=str, default='map',
                        help='Frame ID for the point cloud')
    parser.add_argument('--rate', type=float, default=1.0,
                        help='Publish rate in Hz')
    parsed_args, _ = parser.parse_known_args()

    node = SparseTomogramPublisher(parsed_args.pickle, parsed_args.topic,
                                   parsed_args.frame, parsed_args.rate)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
