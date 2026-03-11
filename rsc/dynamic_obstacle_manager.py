#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from nav_msgs.msg import Odometry
import math

class DynamicObstacle:
    def __init__(self, name, x_range, y_range, speed=0.5, radius=2.0):
        self.name = name
        self.x_center = (x_range[0] + x_range[1]) / 2.0
        self.y_center = (y_range[0] + y_range[1]) / 2.0
        self.radius = radius
        self.speed = speed
        self.angle = 0.0
        self.z = 0.5

    def update_pose(self, dt, robot_pos=None, safety_radius=1.0):
        """Update the internal angle for circular motion, avoiding robot exclusion zone."""
        # Calculate next position
        next_angle = self.angle + self.speed * dt
        if next_angle > 2 * math.pi:
            next_angle -= 2 * math.pi
        
        next_x = self.x_center + self.radius * math.cos(next_angle)
        next_y = self.y_center + self.radius * math.sin(next_angle)
        
        # Check if next position is in robot exclusion zone
        wait_mode = False
        if robot_pos is not None:
            dist_to_robot = math.sqrt(
                (next_x - robot_pos[0])**2 + 
                (next_y - robot_pos[1])**2
            )
            
            # If too close to robot, skip this update (wait)
            # Adding 0.5m buffer for obstacle half-size
            if dist_to_robot < (safety_radius):
                wait_mode = True
                # Don't update angle, effectively "waiting"
                # Return current pose instead
                pose = Pose()
                pose.position.x = self.x_center + self.radius * math.cos(self.angle)
                pose.position.y = self.y_center + self.radius * math.sin(self.angle)
                pose.position.z = self.z
                
                yaw = self.angle + math.pi / 2
                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = sy
                pose.orientation.w = cy
                
                return pose, wait_mode
        
        # Safe to move, update angle
        self.angle = next_angle
        
        pose = Pose()
        pose.position.x = next_x
        pose.position.y = next_y
        pose.position.z = self.z
        
        # Orient towards the tangent direction
        yaw = self.angle + math.pi / 2
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = sy
        pose.orientation.w = cy
        
        return pose, wait_mode

class DynamicObstacleManager(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_manager')
        
        # Robot position tracking
        self.robot_position = None  # [x, y, z]
        self.safety_radius = 1.0  # 1m exclusion zone around robot
        
        # Subscribe to robot odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Wait for service
        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')
        self.get_logger().info("Waiting for /gazebo/set_entity_state service...")
        
        # Initialize 6 boxes (box1 to box6) as defined in the world file
        self.obstacles = [
            DynamicObstacle("box1", (-5.0, -1.0), (-5.0, -1.0), speed=0.8, radius=2.5),
            DynamicObstacle("box2", (1.0, 5.0), (1.0, 5.0), speed=0.6, radius=2.0),
            DynamicObstacle("box3", (-10.0, -6.0), (2.0, 6.0), speed=1.0, radius=2.5),
            DynamicObstacle("box4", (4.0, 8.0), (-8.0, -4.0), speed=0.7, radius=2.8),
            DynamicObstacle("box5", (-15.0, -10.0), (-15.0, -10.0), speed=0.9, radius=3.2),
            DynamicObstacle("box6", (-5.0, -10.0), (0.0, -10.0), speed=0.5, radius=3.0),
        ]

        # Timer for updating obstacle positions (50Hz)
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.update_obstacles)
        
        self.get_logger().info(
            f"Dynamic Obstacle Manager initialized with {len(self.obstacles)} obstacles. "
            f"Robot exclusion zone: {self.safety_radius}m"
        )

    def odom_callback(self, msg):
        """Store robot's current position from odometry."""
        pos = msg.pose.pose.position
        was_none = self.robot_position is None
        self.robot_position = [pos.x, pos.y, pos.z]
        
        if was_none:
            self.get_logger().info(
                f"Robot odometry received. Initial position: "
                f"({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})"
            )

    def update_obstacles(self):
        if not self.set_state_client.service_is_ready():
            return

        waiting_obstacles = []
        for obs in self.obstacles:
            # Pass robot position to update_pose for collision avoidance
            pose, wait_mode = obs.update_pose(
                self.timer_period, 
                robot_pos=self.robot_position,
                safety_radius=self.safety_radius
            )
            
            if wait_mode:
                waiting_obstacles.append(obs.name)
            
            req = SetEntityState.Request()
            req.state.name = obs.name
            req.state.pose = pose
            req.state.reference_frame = "world"
            
            # Use async call to not block the timer
            self.set_state_client.call_async(req)
        
        # Log waiting obstacles periodically
        if waiting_obstacles:
            self.get_logger().info(
                f"Obstacles waiting for robot clearance: {', '.join(waiting_obstacles)}",
                throttle_duration_sec=2.0
            )

def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
