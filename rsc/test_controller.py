import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ContactsState
from pct_planner_msgs.srv import PlanPath as PathSrv
from std_srvs.srv import Empty
import random
import math
import time
import sys
import os
import csv
from datetime import datetime

# Import Managers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from obstacle_manager import ObstacleManager
from dynamic_obstacle_manager import DynamicObstacleManager

class TestController(Node):
    def __init__(self, obstacle_manager):
        super().__init__('test_controller')
        self.obs_manager = obstacle_manager

        
        # Configuration
        self.map_bounds = {'x': (-17.0, -15.0), 'y': (4.0, 5.0)}
        self.goal_tolerance = 1.0  # meters
        self.test_timeout = 60.0   # seconds
        self.collision_detected = False
        self.current_pose = None
        self.goal_pose = None
        self.is_planning = False
        self.start_time = 0.0

        # Clients
        self.plan_client = self.create_client(PathSrv, '/plan_path')
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        
        # Round-trip mode: robot walks back and forth between two fixed waypoints
        self.round_trip = True
        self.round_trip_initialized = False
        self.waypoint_a = None
        self.waypoint_b = None
        self.current_leg = 0  # 0: A→B, 1: B→A

        # Timeout tracking
        self.consecutive_timeouts = 0
        self.consecutive_planning_failures = 0
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            10)
        
        # CSV Logging Setup
        self.csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results.csv')
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Start_X', 'Start_Y', 'Goal_X', 'Goal_Y',
                    'Result', 'Total_Collisions', 'High_Force_Collisions', 'Duration', 'Avg_Speed',
                    'Trunk_Collisions', 'FL_Thigh_Collisions', 'FR_Thigh_Collisions',
                    'RL_Thigh_Collisions', 'RR_Thigh_Collisions',
                    'Final_X', 'Final_Y'
                ])
        
        # Collision Subscribers
        self.collision_subs = []
        collision_topics = [
            '/go2_gazebo/trunk_bumper',
            '/go2_gazebo/FL_thigh_bumper',
            '/go2_gazebo/FR_thigh_bumper',
            '/go2_gazebo/RL_thigh_bumper',
            '/go2_gazebo/RR_thigh_bumper'
            # Add or remove topics as per actual Gazebo plugins
        ]
        
        for topic in collision_topics:
            self.collision_subs.append(
                self.create_subscription(
                    ContactsState, topic,
                    lambda msg, t=topic: self.collision_callback(msg, t), 10)
            )

        self.get_logger().info('Test Controller Initialized')
        
        # Speed tracking
        self.total_distance = 0.0
        self.prev_pose = None

        # Per-joint collision tracking
        self.collision_topics = collision_topics
        self.joint_collision_counts = {topic: 0 for topic in collision_topics}
        self.high_force_collision_count = 0  # Track high-force impacts (force > 100)

        # Control Loop
        self.timer = self.create_timer(0.2, self.control_loop)
        self.state = 'IDLE' # IDLE, SPAWNING, PLANNING, MOVING, FINISHED
        self.collision_count = 0  # Initialize collision counter

    def odom_callback(self, msg):
        new_pose = msg.pose.pose
        # Accumulate distance for average speed calculation
        if self.state == 'MOVING' and self.prev_pose is not None:
            dx = new_pose.position.x - self.prev_pose.position.x
            dy = new_pose.position.y - self.prev_pose.position.y
            self.total_distance += math.sqrt(dx * dx + dy * dy)
        self.prev_pose = new_pose
        self.current_pose = new_pose

    def collision_callback(self, msg, topic):
        # Check if there are any contacts
        if msg.states:
            self.collision_detected = True
            self.collision_count += 1
            # Track per-joint collisions using the known topic
            self.joint_collision_counts[topic] = self.joint_collision_counts.get(topic, 0) + 1
            
            # Check for high-force impacts (force magnitude > 100)
            for state in msg.states:
                if hasattr(state, 'total_wrench') and hasattr(state.total_wrench, 'force'):
                    force = state.total_wrench.force
                    force_magnitude = math.sqrt(force.x**2 + force.y**2 + force.z**2)
                    if force_magnitude > 100.0:
                        self.high_force_collision_count += 1
                        self.get_logger().error(
                            f'HIGH FORCE COLLISION on {topic}! '
                            f'Force magnitude: {force_magnitude:.2f} N '
                            f'(x={force.x:.2f}, y={force.y:.2f}, z={force.z:.2f})'
                        )
                        break  # Only count once per callback
            
            self.get_logger().warn(
                f'Collision on {topic}! '
                f'Joint: {self.joint_collision_counts[topic]}, Total: {self.collision_count}, '
                f'High-Force: {self.high_force_collision_count}'
            )


    def generate_random_point(self):
        x = random.uniform(self.map_bounds['x'][0], self.map_bounds['x'][1])
        y = random.uniform(self.map_bounds['y'][0], self.map_bounds['y'][1])
        # z = 6.0 if (-18.0 <= x <= -12.0 and -15.0 <= y <= -10.0) else 0.0
        z=0.5
        return [x, y, z]

    def request_plan(self, start, end):
        if not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/plan_path service not available')
            return None

        req = PathSrv.Request()
        path_msg = Path()
        path_msg.header.frame_id = "world"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        p1 = PoseStamped()
        p1.pose.position.x = float(start[0])
        p1.pose.position.y = float(start[1])
        p1.pose.position.z = float(start[2])
        path_msg.poses.append(p1)

        p2 = PoseStamped()
        p2.pose.position.x = float(end[0])
        p2.pose.position.y = float(end[1])
        p2.pose.position.z = float(end[2])
        path_msg.poses.append(p2)

        req.path = path_msg
        return self.plan_client.call_async(req)

    def dist(self, p1, p2):
        return math.sqrt((p1.position.x - p2.position.x)**2 + (p1.position.y - p2.position.y)**2)

    def reset_gazebo_world(self):
        """Call Gazebo reset_world service"""
        if not self.reset_world_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/reset_world service not available')
            return False
        
        req = Empty.Request()
        # future = self.reset_world_client.call_async(req)
        # We'll handle this asynchronously, just log for now
        self.get_logger().info('Called /reset_world service to reset Gazebo simulation')
        return True
    
    def save_result(self, result_status):
        if not hasattr(self, 'start_pt') or not self.start_pt or not self.goal_pose:
            return
            
        duration = time.time() - self.start_time
        avg_speed = self.total_distance / duration if duration > 0 else 0.0
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Final position (useful for TIMEOUT)
        final_x = f"{self.current_pose.position.x:.2f}" if self.current_pose else 'N/A'
        final_y = f"{self.current_pose.position.y:.2f}" if self.current_pose else 'N/A'

        # Per-joint collision counts (in topic order)
        joint_counts = [self.joint_collision_counts.get(t, 0) for t in self.collision_topics]

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                f"{self.start_pt[0]:.2f}", f"{self.start_pt[1]:.2f}",
                f"{self.goal_pose.pose.position.x:.2f}", f"{self.goal_pose.pose.position.y:.2f}",
                result_status,
                self.collision_count,
                self.high_force_collision_count,
                f"{duration:.2f}",
                f"{avg_speed:.3f}",
                *joint_counts,
                final_x, final_y
            ])
        self.get_logger().info(
            f"Test result saved to {self.csv_file} | "
            f"Avg Speed: {avg_speed:.3f} m/s | "
            f"High-Force Collisions: {self.high_force_collision_count}"
        )

    def control_loop(self):
        if self.state == 'IDLE':
            if self.current_pose is None:
                self.get_logger().info('Waiting for /odom...', throttle_duration_sec=2.0)
                return
            
            self.get_logger().info('Starting new test cycle...')

            self.state = 'PLANNING'
            
        
            
        elif self.state == 'PLANNING':
            # Setup Goal
            if self.round_trip:
                if not self.round_trip_initialized:
                    # First run: use current position as waypoint A, random point as waypoint B
                    self.waypoint_a = [
                        self.current_pose.position.x,
                        self.current_pose.position.y,
                        self.current_pose.position.z
                    ]
                    self.waypoint_b = self.generate_random_point()
                    self.round_trip_initialized = True
                    self.current_leg = 0
                    self.get_logger().info(
                        f'Round-trip initialized: '
                        f'A=({self.waypoint_a[0]:.2f}, {self.waypoint_a[1]:.2f}), '
                        f'B=({self.waypoint_b[0]:.2f}, {self.waypoint_b[1]:.2f})'
                    )
                # Select start/end based on current leg
                if self.current_leg == 0:
                    self.start_pt = list(self.waypoint_a)
                    end_pt = list(self.waypoint_b)
                else:
                    self.start_pt = list(self.waypoint_b)
                    end_pt = list(self.waypoint_a)
            else:
                self.start_pt = [self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z]
                end_pt = self.generate_random_point()

            self.goal_pose = PoseStamped()
            self.goal_pose.pose.position.x = end_pt[0]
            self.goal_pose.pose.position.y = end_pt[1]
            self.goal_pose.pose.position.z = end_pt[2]

            leg_label = f'Leg {self.current_leg} (A→B)' if self.current_leg == 0 else f'Leg {self.current_leg} (B→A)'
            self.get_logger().info(
                f'[{leg_label if self.round_trip else "Random"}] '
                f'Planning from ({self.start_pt[0]:.2f}, {self.start_pt[1]:.2f}) '
                f'to ({end_pt[0]:.2f}, {end_pt[1]:.2f})'
            )
            
            self.plan_future = self.request_plan(self.start_pt, end_pt)
            
            if self.plan_future:
               self.state = 'SPAWNING' # Directly move to SPAWNING state, assuming planner triggers movement
               self.start_time = time.time()
               self.collision_detected = False
               self.collision_count = 0
               self.high_force_collision_count = 0
               self.total_distance = 0.0
               self.prev_pose = self.current_pose
               self.joint_collision_counts = {t: 0 for t in self.collision_topics}
               self.get_logger().info('Plan requested. Entering SPAWNING state to monitor...')
            else:
               self.get_logger().error('Failed to call plan service. Retrying...')
               self.state = 'IDLE'
        elif self.state == 'SPAWNING':
            # Manage Obstacles - only spawn new obstacles (clearing happens automatically every 10)
            # Spawn new static obstacle with a base name
            success = self.obs_manager.spawn_static_obstacle('obstacle', (self.start_pt[0]+1.0, self.goal_pose.pose.position.x), (self.start_pt[1]+1.0, self.goal_pose.pose.position.y))
            
            if not success:
                self.get_logger().warn('Failed to spawn obstacle, but continuing...')
            
            # Spawn new dynamic (if implemented)
            # self.obs_manager.spawn_dynamic_obstacle('dynamic', ...)
            self.get_logger().info('Obstacle spawned. Entering MOVING state to monitor...')

            self.state = 'MOVING'
        elif self.state == 'MOVING':
            # Check for plan result in background if needed, but we don't block for it now
            if self.plan_future and self.plan_future.done():
                try:
                    resp = self.plan_future.result()
                    if resp.success:
                         # Plan success logic if needed
                         self.consecutive_planning_failures = 0  # Reset on successful planning
                    else:
                        self.consecutive_planning_failures += 1
                        self.get_logger().error(f'Plan service returned failure. Consecutive failures: {self.consecutive_planning_failures}')
                        self.save_result('PLANNING_FAILED')
                        
                        # Reset Gazebo after 5 consecutive planning failures
                        if self.consecutive_planning_failures >= 5:
                            self.get_logger().warn('Five consecutive planning failures detected! Resetting Gazebo world...')
                            if self.reset_gazebo_world():
                                self.consecutive_planning_failures = 0  # Reset counter after successful reset
                        
                        self.state = 'FINISHED'
                except Exception as e:
                     self.get_logger().error(f'Plan Service Exception: {e}')
                     self.consecutive_planning_failures += 1
                     self.save_result('PLANNING_EXCEPTION')
                     self.state = 'FINISHED'
                finally:
                    self.plan_future = None # Stop checking

            # Check Timeout
            if time.time() - self.start_time > self.test_timeout:
                self.consecutive_timeouts += 1
                self.get_logger().error(
                    f'Test Finished: TIMEOUT. '
                    f'Collisions: {self.collision_count} (High-Force: {self.high_force_collision_count}). '
                    f'Consecutive Timeouts: {self.consecutive_timeouts}'
                )
                self.save_result('TIMEOUT')
                
                # Reset Gazebo after 2 consecutive timeouts
                if self.consecutive_timeouts >= 2:
                    self.get_logger().warn('Two consecutive timeouts detected! Resetting Gazebo world...')
                    if self.reset_gazebo_world():
                        self.consecutive_timeouts = 0  # Reset counter after successful reset
                
                self.state = 'FINISHED'
                return

            # Check Goal
            if self.current_pose and self.goal_pose:
                d = self.dist(self.current_pose, self.goal_pose.pose)
                if d < self.goal_tolerance:
                    self.consecutive_timeouts = 0  # Reset timeout counter on success
                    self.consecutive_planning_failures = 0  # Reset planning failure counter on success
                    self.get_logger().info(
                        f'Test Passed: GOAL REACHED. '
                        f'Collisions: {self.collision_count} (High-Force: {self.high_force_collision_count})'
                    )
                    self.save_result('SUCCESS')
                    if self.round_trip:
                        # Flip leg direction for the return journey
                        self.current_leg = 1 - self.current_leg
                        self.get_logger().info(
                            f'Round-trip: switching to '
                            f'{"A→B" if self.current_leg == 0 else "B→A"}'
                        )
                    self.state = 'FINISHED'
        
        elif self.state == 'FINISHED':
             self.get_logger().info('Resetting environment in 3s...')
             # Wait a bit before next run
             # In a async timer callback we cannot sleep easily blocking the thread, 
             # but here we just transition state next time
             self.state = 'IDLE'

def main(args=None):
    rclpy.init(args=args)
    
    obs_manager = ObstacleManager()
    dyn_obs_manager = DynamicObstacleManager()
    test_controller = TestController(obs_manager)

    executor = MultiThreadedExecutor()
    executor.add_node(obs_manager)
    executor.add_node(dyn_obs_manager)
    executor.add_node(test_controller)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        obs_manager.destroy_node()
        dyn_obs_manager.destroy_node()
        test_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
