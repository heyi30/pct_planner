#!/usr/bin/python3
import os
import sys
import time
import pickle
import numpy as np
from numpy.lib import recfunctions as rfn
import cupy as cp
import gc
import subprocess
import argparse
from array import array

import open3d as o3d
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, ByteMultiArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from tomogram import Tomogram

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import POINT_FIELDS_XYZI, GRID_POINTS_XYZI
from config import Config
from config.scene import ScenePCD

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class Tomography(Node):
    def __init__(self, cfg, scene_cfg):
        super().__init__('pointcloud_tomography')
        self.cfg = cfg
        self.scene_cfg = scene_cfg
        self.map_frame = cfg.ros.map_frame
        self.export_dir = rsg_root + cfg.map.export_dir
        self.resolution = scene_cfg.map.resolution
        self.ground_h = scene_cfg.map.ground_h
        self.slice_dh = scene_cfg.map.slice_dh

        self.center = np.zeros(2, dtype=np.float32)
        self.tomogram = Tomogram(scene_cfg)
        self.is_initialized = False

        tomogram_topic = cfg.ros.tomogram_topic
        self.tomogram_pub = self.create_publisher(PointCloud2, tomogram_topic, 10)
        self.tomogram_data_pub = self.create_publisher(ByteMultiArray, '/tomogram_data', 10)

        self.load_and_process_pcd()

    def load_and_process_pcd(self):
        if self.is_initialized:
            return

        pcd_file_name = getattr(self.scene_cfg.pcd, 'file_name', 'test.pcd')
        pcd_file_path = os.path.join(rsg_root, 'rsc', 'pcd', pcd_file_name)

        self.get_logger().info(f"Loading PCD file from: {pcd_file_path}")

        try:
            pcd = o3d.io.read_point_cloud(pcd_file_path)
            points = np.asarray(pcd.points, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Failed to read PCD file: {e}")
            return

        if points.size == 0:
            self.get_logger().warn("Loaded an empty point cloud. Skipping processing.")
            return

        self.get_logger().info("PCD points: %d" % points.shape[0])
        self.points_max = np.max(points, axis=0)
        self.points_min = np.min(points, axis=0)           
        self.points_min[-1] = self.ground_h
        self.map_dim_x = int(np.ceil((self.points_max[0] - self.points_min[0]) / self.resolution)) + 4
        self.map_dim_y = int(np.ceil((self.points_max[1] - self.points_min[1]) / self.resolution)) + 4
        n_slice_init = int(np.ceil((self.points_max[2] - self.points_min[2]) / self.slice_dh))
        self.center = (self.points_max[:2] + self.points_min[:2]) / 2
        self.slice_h0 = self.points_min[-1] + self.slice_dh
        self.tomogram.initMappingEnv(self.center, self.map_dim_x, self.map_dim_y, n_slice_init, self.slice_h0)

        self.get_logger().info("Map center: [%.2f, %.2f]" % (self.center[0], self.center[1]))
        self.get_logger().info("Dim_x: %d" % self.map_dim_x)
        self.get_logger().info("Dim_y: %d" % self.map_dim_y)
        self.get_logger().info("Num slices init: %d" % n_slice_init)

        self.VISPROTO_I, self.VISPROTO_P = \
            GRID_POINTS_XYZI(self.resolution, self.map_dim_x, self.map_dim_y)

        self.process(points)
        self.is_initialized = True
        self.get_logger().info("Tomography initialized and processed first point cloud.")

        
    def process(self, points):        
        t_map = 0.0
        t_trav = 0.0
        t_simp = 0.0
        t_all = 0.0
        n_repeat = 10

        """ 
        GPU time benchmark, where CUDA events are synchronized for correct time measurement.
        The function is repeatedly run for n_repeat times to calculate the average processing time of each modules.
        The time of the first warm-up run is excluded to reduce timing fluctuation and exclude the overhead in initial invocations.
        See https://docs.cupy.dev/en/stable/user_guide/performance.html for more details
        """

        t_start = time.time()
        layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, slice_heights, t_gpu = self.tomogram.point2map(points)


        self.get_logger().info("Num slices simp: %d" % layers_g.shape[0])
        self.get_logger().info("Num repeats (for benchmarking only): %d" % n_repeat)
        self.get_logger().info(" -- avg t_map  (ms): %f" % (t_map / n_repeat))
        self.get_logger().info(" -- avg t_trav (ms): %f" % (t_trav / n_repeat))
        self.get_logger().info(" -- avg t_simp (ms): %f" % (t_simp / n_repeat))
        self.get_logger().info(" -- avg t_all  (ms): %f" % (t_all / n_repeat))

        self.n_slice = layers_g.shape[0]

        map_file = "scene_map"
        tomogram_data = np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c))
        
        data_dict = self.exportTomogram(tomogram_data, map_file, slice_heights)
        self.publish_tomogram_data(data_dict)
        self.publishTomogram(layers_g, layers_t)

    def exportTomogram(self, tomogram, map_file, slice_heights):        
        data_dict = {
            'data': tomogram.astype(np.float16),
            'resolution': self.resolution,
            'center': self.center,
            'slice_h0': self.slice_h0,
            'slice_dh': self.slice_dh,
            'slice_heights': slice_heights.astype(np.float16),
        }
        file_name = map_file + '.pickle'
        with open(self.export_dir + file_name, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.get_logger().info("Tomogram exported: %s" % file_name)
        return data_dict

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
            layer_points[:, 2] = vis_g[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            layer_points[:, 3] = vis_t[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            if global_points is None:
                global_points = valid_points
            else:
                global_points = np.concatenate((global_points, valid_points), axis=0)

        layer_points[:, 2] = vis_g[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
        layer_points[:, 3] = vis_t[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
        valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
        global_points = np.concatenate((global_points, valid_points), axis=0)
        
        points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, global_points)
        self.tomogram_pub.publish(points_msg)
        self.get_logger().info("Tomogram visualization published to /tomogram")

        
    def process(self, points):        
        t_map = 0.0
        t_trav = 0.0
        t_simp = 0.0
        t_all = 0.0
        n_repeat = 10

        """ 
        GPU time benchmark, where CUDA events are synchronized for correct time measurement.
        The function is repeatedly run for n_repeat times to calculate the average processing time of each modules.
        The time of the first warm-up run is excluded to reduce timing fluctuation and exclude the overhead in initial invocations.
        See https://docs.cupy.dev/en/stable/user_guide/performance.html for more details
        """
        for i in range(n_repeat + 1):
            t_start = time.time()
            layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, slice_heights, t_gpu = self.tomogram.point2map(points)

            if i > 0:
                t_map += t_gpu['t_map']
                t_trav += t_gpu['t_trav']
                t_simp += t_gpu['t_simp']
                t_all += (time.time() - t_start) * 1e3

        self.get_logger().info("Num slices simp: %d" % layers_g.shape[0])
        self.get_logger().info("Num repeats (for benchmarking only): %d" % n_repeat)
        self.get_logger().info(" -- avg t_map  (ms): %f" % (t_map / n_repeat))
        self.get_logger().info(" -- avg t_trav (ms): %f" % (t_trav / n_repeat))
        self.get_logger().info(" -- avg t_simp (ms): %f" % (t_simp / n_repeat))
        self.get_logger().info(" -- avg t_all  (ms): %f" % (t_all / n_repeat))

        self.n_slice = layers_g.shape[0]

        map_file = "scene_map"
        tomogram_data = np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c))
        
        data_dict = self.exportTomogram(tomogram_data, map_file, slice_heights)
        self.publish_tomogram_data(data_dict)
        self.publishTomogram(layers_g, layers_t)

    def exportTomogram(self, tomogram, map_file, slice_heights):        
        data_dict = {
            'data': tomogram.astype(np.float16),
            'resolution': self.resolution,
            'center': self.center,
            'slice_h0': self.slice_h0,
            'slice_dh': self.slice_dh,
            'slice_heights': slice_heights.astype(np.float16),
        }
        file_name = map_file + '.pickle'
        with open(self.export_dir + file_name, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.get_logger().info("Tomogram exported: %s" % file_name)
        return data_dict

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
            layer_points[:, 2] = vis_g[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            layer_points[:, 3] = vis_t[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            if global_points is None:
                global_points = valid_points
            else:
                global_points = np.concatenate((global_points, valid_points), axis=0)

        layer_points[:, 2] = vis_g[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
        layer_points[:, 3] = vis_t[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
        valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
        global_points = np.concatenate((global_points, valid_points), axis=0)
        
        points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, global_points)
        self.tomogram_pub.publish(points_msg)
        self.get_logger().info("Tomogram visualization published to /tomogram")


def main(args=None):
    # Launch rviz2
    rclpy.logging.get_logger('tomography').info("Launching rviz2...")
    rviz_config_path = rsg_root + '/rsc/rviz/pct_ros.rviz'
    subprocess.Popen(['rviz2', '-d', rviz_config_path])

    # Clean up GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    print(f"[INFO] GPU memory usage: {cp.get_default_memory_pool().used_bytes()/1024**3:.2f} GB used")

    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='Map', help='Name of the scene. Available: [\'Map\']')
    parser.add_argument('--pcd', type=str, default=None, help='PCD file name under rsc/pcd/. Overrides scene config if set.')
    ros_args, _ = parser.parse_known_args()

    cfg = Config()
    scene_cfg = getattr(__import__('config'), 'Scene' + ros_args.scene)()

    if ros_args.pcd is not None:
        if not hasattr(scene_cfg, 'pcd'):
            scene_cfg.pcd = ScenePCD()
        scene_cfg.pcd.file_name = ros_args.pcd

    mapping_node = Tomography(cfg, scene_cfg)

    try:
        rclpy.spin(mapping_node)
    except KeyboardInterrupt:
        pass
    finally:
        mapping_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
