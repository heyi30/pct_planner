import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import time

class WaypointsPublisher(Node):
    def __init__(self):
        super().__init__('waypoints_publisher')
        self.waypoints_pub = self.create_publisher(Path, '/waypoints', 1)
        self.get_logger().info('Waypoints Publisher Node has been started.')

    def publish_waypoints(self, points):
        path_msg = Path()
        path_msg.header.frame_id = "world"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for pt in points:
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            pose.pose.position.z = float(pt[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.waypoints_pub.publish(path_msg)
        self.get_logger().info(f'Published {len(points)} waypoints to /waypoints')

def main(args=None):
    rclpy.init(args=args)
    node = WaypointsPublisher()

    # Example waypoints
    pts = [
        # [0.0, 0.0, 0.0],
        # [5.0, -6.0, 0.0],
        # [5.7, 7.3, 0.0],
        # [-15.6, -13.5, 6.0]
        [-15.6, -13.5, 5.5],
        [5.7, 7.3, 0.0],
        [5.0, -6.0, 0.0],
        [0.0, 0.0, 0.0]
    ]

    floor_pts=[
        [0.321486, -0.504209, 0],
        [1.2041, -14.4183, 0],
        [-6.13168, 7.34961, 0]

    ]
    shangfei_pts=[
        [-43.3876, -13.932, 0],
        [-46.5037, -12.6024, 0],
        [-46.234, 7.21439, 0],
        [-43.9995, 6.96735, 0],  
        [-43.3843, 3.58991, 0],
        [-40.9803, 3.10264, 0],
        [-37.8602, 3.93038, 0],
        [-37.1889, 6.11121, 0],
        [-31.7,5.93,2.0]
    ]
    
    time.sleep(1)  # Give some time for publishers to set up
    node.publish_waypoints(shangfei_pts)
    time.sleep(1) 

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
