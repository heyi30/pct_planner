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

import open3d as o3d
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from tomogram import Tomogram

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import POINT_FIELDS_XYZI, GRID_POINTS_XYZI
from config import Config
from config.scene import ScenePCD

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'

# Sparse map parameters. These must stay in sync with the values used by
# SparseTomogramPlanner / SparseAstar.
SPARSE_ASTAR_COST_THRESHOLD = 35.0
SPARSE_GATEWAY_COST_DELTA = 8.0
SPARSE_GATEWAY_HEIGHT_DELTA = 0.1


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

        self.publishTomogram(layers_g, layers_t)
        self.exportSparseTomogram(layers_t, layers_g, layers_c, slice_heights)

    @staticmethod
    def _compute_gateway(trav, elev_g):
        """Gateway logic moved from planner_wrapper.py to the sparse export stage."""
        diff_t = trav[1:] - trav[:-1]
        diff_g = np.abs(elev_g[1:] - elev_g[:-1])

        gateway_up = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t < -SPARSE_GATEWAY_COST_DELTA
        mask_g = (diff_g < SPARSE_GATEWAY_HEIGHT_DELTA) & np.isfinite(elev_g[1:])
        gateway_up[:-1] = mask_t & mask_g

        gateway_dn = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t > SPARSE_GATEWAY_COST_DELTA
        mask_g = (diff_g < SPARSE_GATEWAY_HEIGHT_DELTA) & np.isfinite(elev_g[:-1])
        gateway_dn[1:] = mask_t & mask_g

        gateway = np.zeros_like(trav, dtype=np.int32)
        gateway[gateway_up] = 2
        gateway[gateway_dn] = -2
        return gateway

    @staticmethod
    def _filter_largest_component(shape, indices, *arrays):
        """Keep only the largest connected component of sparse nodes.

        Uses 26-connectivity in 3-D (layer, row, col).  This removes small
        disconnected islands that the planner cannot reach anyway.
        """
        from scipy import ndimage

        mask = np.zeros(shape, dtype=bool)
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        structure = ndimage.generate_binary_structure(3, 3)
        labeled, n = ndimage.label(mask, structure=structure)
        if n <= 1:
            return (indices,) + arrays

        component_sizes = np.bincount(labeled.ravel())[1:]
        largest_label = int(component_sizes.argmax()) + 1
        keep = labeled[indices[:, 0], indices[:, 1], indices[:, 2]] == largest_label
        if not np.any(keep):
            return (indices,) + arrays, n
        return tuple(arr[keep] for arr in (indices,) + arrays), n

    def exportSparseTomogram(self, layers_t, layers_g, layers_c, slice_heights):
        """Export the sparse tomogram used by SparseTomogramPlanner / SparseAstar."""
        t0 = time.time()

        gateway = self._compute_gateway(layers_t, layers_g)

        valid_height = np.isfinite(layers_g)
        valid_cost = np.isfinite(layers_t)
        traversable = layers_t <= SPARSE_ASTAR_COST_THRESHOLD
        gateway_keep = gateway != 0

        sparse_mask = (
            valid_height
            & valid_cost
            & traversable
        ) | gateway_keep

        coords = np.argwhere(sparse_mask)
        # tomogram arrays are indexed [layer, x, y]; sparse format uses [layer, row=y, col=x]
        indices = np.stack([coords[:, 0], coords[:, 2], coords[:, 1]], axis=1)
        indices = indices.astype(np.int32)

        (indices, trav_vals, elev_g_vals, elev_c_vals, gateway_vals), n_components = \
            self._filter_largest_component(
                [int(self.n_slice), int(self.map_dim_y), int(self.map_dim_x)],
                indices,
                layers_t[sparse_mask].astype(np.float32),
                layers_g[sparse_mask].astype(np.float32),
                layers_c[sparse_mask].astype(np.float32),
                gateway[sparse_mask].astype(np.int32),
            )

        data_dict = {
            'format': 'tomogram_sparse_v1',
            'shape': [int(self.n_slice), int(self.map_dim_y), int(self.map_dim_x)],
            'resolution': float(self.resolution),
            'center': self.center.astype(np.float64),
            'slice_h0': float(self.slice_h0),
            'slice_dh': float(self.slice_dh),
            'slice_heights': slice_heights.astype(np.float32),
            'indices': indices,
            'trav': trav_vals,
            'elev_g': elev_g_vals,
            'elev_c': elev_c_vals,
            'gateway': gateway_vals,
        }

        file_name = 'scene_map_sparse.pickle'
        file_path = self.export_dir + file_name
        with open(file_path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        dense_cells = int(self.n_slice) * int(self.map_dim_y) * int(self.map_dim_x)
        sparse_nodes = int(indices.shape[0])
        sparse_ratio = sparse_nodes / dense_cells if dense_cells > 0 else 0.0
        elapsed_ms = (time.time() - t0) * 1e3

        self.get_logger().info("Sparse tomogram exported: %s" % file_name)
        self.get_logger().info("  dense_cells = %d" % dense_cells)
        self.get_logger().info("  sparse_nodes = %d" % sparse_nodes)
        self.get_logger().info("  components = %d (kept largest, removed %d nodes)" %
                               (n_components, int(coords.shape[0]) - sparse_nodes))
        self.get_logger().info("  sparse_ratio = %.4f" % sparse_ratio)
        self.get_logger().info("  file_size = %.2f MB" % (os.path.getsize(file_path) / (1024.0 * 1024.0)))
        self.get_logger().info("  export_time = %.2f ms" % elapsed_ms)

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
