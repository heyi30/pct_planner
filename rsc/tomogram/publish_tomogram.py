#!/usr/bin/python3
import os
import sys
import pickle
import argparse
import subprocess

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, ByteMultiArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

# Reuse the same config definitions as tomography.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tomography'))
from config import POINT_FIELDS_XYZI, GRID_POINTS_XYZI
from config import Config

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class TomogramPublisher(Node):
    def __init__(self, cfg, pickle_path):
        super().__init__('pointcloud_tomography')
        self.cfg = cfg
        self.map_frame = cfg.ros.map_frame
        self.resolution = None
        self.slice_dh = None
        self.center = None
        self.slice_h0 = None

        self.tomogram_pub = self.create_publisher(PointCloud2, cfg.ros.tomogram_topic, 10)
        self.tomogram_data_pub = self.create_publisher(ByteMultiArray, '/tomogram_data', 10)

        self.load_pickle(pickle_path)
        self.publish_data()

    def load_pickle(self, pickle_path):
        self.get_logger().info(f"Loading tomogram pickle from: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            data_dict = pickle.load(f)

        self.data_dict = data_dict
        tomogram = data_dict['data'].astype(np.float32)  # (5, N_slice, dim_y, dim_x)
        self.resolution = float(data_dict['resolution'])
        self.center = np.asarray(data_dict['center'], dtype=np.float32)
        self.slice_h0 = float(data_dict['slice_h0'])
        self.slice_dh = float(data_dict['slice_dh'])
        self.slice_heights = np.asarray(data_dict['slice_heights'], dtype=np.float32)

        self.layers_t = tomogram[0]
        self.layers_g = tomogram[3]

        dim_y, dim_x = self.layers_g.shape[1], self.layers_g.shape[2]
        self.map_dim_x = dim_x
        self.map_dim_y = dim_y
        self.VISPROTO_I, self.VISPROTO_P = GRID_POINTS_XYZI(self.resolution, dim_x, dim_y)

        self.get_logger().info("Map center: [%.2f, %.2f]" % (self.center[0], self.center[1]))
        self.get_logger().info("Dim_x: %d" % dim_x)
        self.get_logger().info("Dim_y: %d" % dim_y)
        self.get_logger().info("Num slices: %d" % self.layers_g.shape[0])

    def publish_data(self):
        self.publish_tomogram_data(self.data_dict)
        self.publishTomogram(self.layers_g, self.layers_t)

    def publish_tomogram_data(self, data_dict):
        pickled_data = pickle.dumps(data_dict)
        msg = ByteMultiArray()
        msg.data = [bytes([i]) for i in pickled_data]
        self.tomogram_data_pub.publish(msg)
        self.get_logger().info("Tomogram data published to /tomogram_data")

    def publishTomogram(self, layers_g, layers_t):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame

        n_slice = layers_g.shape[0]
        vis_g = layers_g.copy()
        vis_t = layers_t.copy()
        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.center

        global_points = None
        for i in range(n_slice - 1):
            mask_h = (vis_g[i + 1] - vis_g[i]) < self.slice_dh
            vis_g[i, mask_h] = np.nan
            vis_t[i + 1, mask_h] = np.minimum(vis_t[i, mask_h], vis_t[i + 1, mask_h])
            layer_points[:, 2] = vis_g[i, self.VISPROTO_I[:, 1], self.VISPROTO_I[:, 0]]
            layer_points[:, 3] = vis_t[i, self.VISPROTO_I[:, 1], self.VISPROTO_I[:, 0]]
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            if global_points is None:
                global_points = valid_points
            else:
                global_points = np.concatenate((global_points, valid_points), axis=0)

        layer_points[:, 2] = vis_g[-1, self.VISPROTO_I[:, 1], self.VISPROTO_I[:, 0]]
        layer_points[:, 3] = vis_t[-1, self.VISPROTO_I[:, 1], self.VISPROTO_I[:, 0]]
        valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
        global_points = np.concatenate((global_points, valid_points), axis=0)

        points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, global_points)
        self.tomogram_pub.publish(points_msg)
        self.get_logger().info("Tomogram visualization published to /tomogram")


def main(args=None):
    rclpy.logging.get_logger('tomography').info("Launching rviz2...")
    rviz_config_path = rsg_root + '/rsc/rviz/pct_ros.rviz'
    subprocess.Popen(['rviz2', '-d', rviz_config_path])

    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str, default='scene_map.pickle',
                        help='Pickle file name under rsc/tomogram/.')
    parsed_args, _ = parser.parse_known_args()

    cfg = Config()
    pickle_path = os.path.join(rsg_root, 'rsc', 'tomogram', parsed_args.pickle)

    node = TomogramPublisher(cfg, pickle_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
