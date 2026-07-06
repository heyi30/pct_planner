import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from pct_planner_msgs.srv import PlanPath as PathSrv
import time

class WaypointsPublisher(Node):
    def __init__(self):
        super().__init__('waypoints_publisher')
        self.waypoints_pub = self.create_publisher(Path, '/global_trajectory', 10)
        self.get_logger().info('Waypoints Publisher Node has been started.')
        self.clent_=self.create_client(PathSrv,'/plan_path')
    def publish_waypoints(self, points):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for pt in points:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            pose.pose.position.z = float(pt[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        # self.clent_.wait_for_service()
        # request = PathSrv.Request()
        # request.path = path_msg
        # future = self.clent_.call_async(request)
        # rclpy.spin_until_future_complete(self,future)
        # response = future.result()
        # if response.success:
        #     self.get_logger().info(f'Received planned path with {len(response.path.poses)} waypoints from /plan_path service.')
        # else:
        #     self.get_logger().error('Failed to plan path via /plan_path service.')
        self.waypoints_pub.publish(path_msg)
        self.get_logger().info(f'Published {len(points)} waypoints to /waypoints')

def main(args=None):
    rclpy.init(args=args)
    node = WaypointsPublisher()

    # Example waypoints
    pts = [
        # [0.0, 0.0, 0.0],
        # [5.0, -6.0, 0.0],
        [-0.0, 0.5, 0.0],
        [0.3,-6.5,6.0]
        # [0.3,-6.5,6.0],
        # [-0.0, 0.5, 0.0]
        # [5.7, 7.3, 0.0],
        # [5.0, -6.0, 0.0],
        # [0.0, 0.0, 0.0]
    ]

    floor_pts=[
        [0.321486, -0.504209, 0],
        [1.2041, -14.4183, 3.0],
        [-6.13168, 7.34961, 6.0]

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
    slope_pts=[[1.0,0.0,0.5],
                [3.0,0.0,1.0],
                [6.0,0.0,2.0]]
    stair_pts=[[-2.0,5.0,0.5],
                [0.0,5.0,0.5],
                [3.0,5.0,2.0],
                [5.0,5.0,3.0],
                [7.0,5.0,4.0]]
    corridor_pts=[[-2.0,10.0,0.5],
                    [1.0,10.0,0.5],
                    [2.0,10.0,0.5],
                    [3.0,10.0,0.5],
                    [4.0,10.0,0.5],
                    [5.0,10.0,0.5],
                    [6.0,10.0,0.5],
                    [7.0,10.0,0.5],
                    [8.0,10.0,0.5],
                    [9.0,10.0,0.5],
                    [10.0,10.0,0.5]]
    
    # time.sleep(1)  # Give some time for publishers to set up
    node.publish_waypoints(floor_pts)
    # time.sleep(1) 

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
