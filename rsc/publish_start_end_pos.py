import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import time

class StartEndPosPublisher(Node):
    def __init__(self):
        super().__init__('start_end_pos_publisher')
        self.start_pos_pub = self.create_publisher(Point, '/start_pos', 10)
        self.end_pos_pub = self.create_publisher(Point, '/end_pos', 10)
        self.get_logger().info('Start/End Position Publisher Node has been started.')

    def publish_positions(self, start_x, start_y, start_z, end_x, end_y, end_z):
        start_point = Point()
        start_point.x = float(start_x)
        start_point.y = float(start_y)
        start_point.z = float(start_z)

        end_point = Point()
        end_point.x = float(end_x)
        end_point.y = float(end_y)
        end_point.z = float(end_z)

        self.start_pos_pub.publish(start_point)
        self.end_pos_pub.publish(end_point)
        self.get_logger().info(f'Published start_pos: ({start_point.x}, {start_point.y}, {start_point.z})')
        self.get_logger().info(f'Published end_pos: ({end_point.x}, {end_point.y}, {end_point.z})')

def main(args=None):
    rclpy.init(args=args)
    node = StartEndPosPublisher()

    # Example positions (you can change these)
    # These values are taken from the original hardcoded values in plan.py
    # start_x, start_y, start_z = -7.75, 71.84, -0.68
    # end_x, end_y, end_z = 17.11, -9.012, 9.456
    start_x, start_y, start_z = 0, 0, 0
    end_x, end_y, end_z = 1, 6, 0

    # Publish once and then spin to keep the node alive for a short period
    # to ensure message delivery.
    node.publish_positions(start_x, start_y, start_z, end_x, end_y, end_z)
    
    # Keep the node alive for a few seconds to ensure messages are processed
    # by subscribers.
    time.sleep(1) 

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
