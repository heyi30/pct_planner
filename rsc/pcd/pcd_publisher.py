#!/usr/bin/python3
import os
import sys
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'

class PCDPublisher(Node):
    def __init__(self):
        super().__init__('pcd_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, '/global_points', 10)
        timer_period = 10  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        pcd_file = os.path.join(rsg_root, 'rsc', 'pcd', 'dshp.pcd')
        self.get_logger().info(f"Loading PCD file from: {pcd_file}")
        
        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points, dtype=np.float32)
            self.get_logger().info(f"Loaded {len(points)} points from PCD file.")
            
            # Downsample if point count exceeds 1,000,000
            if len(points) > 1000000:
                self.get_logger().info(f"Point count exceeds 1,000,000. Downsampling with voxel size 0.1m...")
                pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.1)
                points = np.asarray(pcd_downsampled.points, dtype=np.float32)
                self.get_logger().info(f"Downsampled to {len(points)} points.")
                
                # Save downsampled point cloud
                downsampled_file = pcd_file.replace('.pcd', '_downsampled.pcd')
                o3d.io.write_point_cloud(downsampled_file, pcd_downsampled)
                self.get_logger().info(f"Saved downsampled point cloud to: {downsampled_file}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to read PCD file: {e}")
            points = []

        if len(points) > 0:
            # The rviz config expects an intensity field. Let's add a dummy one.
            # Create a structured numpy array with x, y, z, intensity fields
            structured_points = np.zeros(points.shape[0], dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('intensity', np.float32)
            ])
            structured_points['x'] = points[:, 0]
            structured_points['y'] = points[:, 1]
            structured_points['z'] = points[:, 2]
            structured_points['intensity'] = 0.0 # Dummy intensity

            header = Header()
            header.frame_id = 'map'
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            self.point_cloud_msg = pc2.create_cloud(header, fields, structured_points)
        else:
            self.point_cloud_msg = None


    def timer_callback(self):
        if self.point_cloud_msg:
            self.point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(self.point_cloud_msg)
            self.get_logger().info('Publishing point cloud')

def main(args=None):
    rclpy.init(args=args)
    pcd_publisher = PCDPublisher()
    try:
        rclpy.spin(pcd_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        pcd_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
