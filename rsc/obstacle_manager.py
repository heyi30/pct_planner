import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
import random
import math
import time

class ObstacleManager(Node):
    def __init__(self):
        super().__init__('obstacle_manager')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.obstacles = []  # List of obstacle names
        self.obstacle_counter = 0  # Unique ID counter for obstacle names
        self.total_spawned = 0  # Total count of spawned obstacles

    def generate_random_pose(self, x_range, y_range):
        pose = Pose()
        pose.position.x = random.uniform(*x_range)
        pose.position.y = random.uniform(*y_range)
        pose.position.z = 0.5  # Assume obstacle is on ground
        return pose

    def spawn_static_obstacle(self, base_name, x_range, y_range):
        """Spawns a simple box or cylinder as static obstacle"""
        # Check if we need to clear obstacles (every 10 obstacles)
        if self.total_spawned > 0 and self.total_spawned % 10 == 0:
            self.get_logger().info(f'Reached {self.total_spawned} obstacles. Clearing all obstacles...')
            self.clear_obstacles()
            time.sleep(0.5)  # Give Gazebo time to process deletions
        
        if not self.spawn_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Spawn service not available')
            return False

        # Generate unique name with counter to avoid conflicts
        self.obstacle_counter += 1
        unique_name = f"{base_name}_{self.obstacle_counter}_{int(time.time() * 1000) % 10000}"
        
        max_attempts = 10
        for attempt in range(max_attempts):
            pose = self.generate_random_pose(x_range, y_range)
            
            # Simple SDF for a unit box with unique link name
            sdf_xml = f"""
            <?xml version='1.0'?>
            <sdf version='1.6'>
              <model name='{unique_name}'>
                <static>true</static>
                <link name='link_{self.obstacle_counter}'>
                  <collision name='collision'>
                    <geometry>
                      <box>
                        <size>0.5 0.5 1.0</size>
                      </box>
                    </geometry>
                  </collision>
                  <visual name='visual'>
                    <geometry>
                      <box>
                        <size>0.5 0.5 1.0</size>
                      </box>
                    </geometry>
                    <material>
                      <script>
                        <uri>file://media/materials/scripts/gazebo.material</uri>
                        <name>Gazebo/Red</name>
                      </script>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>
            """

            req = SpawnEntity.Request()
            req.name = unique_name
            req.xml = sdf_xml
            req.initial_pose = pose

            future = self.spawn_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() and future.result().success:
                self.obstacles.append(unique_name)
                self.total_spawned += 1
                self.get_logger().info(f"Spawned static obstacle: {unique_name} at ({pose.position.x:.2f}, {pose.position.y:.2f}) [Total: {self.total_spawned}, Current: {len(self.obstacles)}]")
                return True
            else:
                error_msg = future.result().status_message if future.result() else "Unknown error"
                self.get_logger().warn(f"Failed to spawn {unique_name}: {error_msg}, attempt {attempt + 1}/{max_attempts}")
                time.sleep(0.1)  # Brief delay between attempts
        
        self.get_logger().error(f"Failed to spawn {unique_name} after {max_attempts} attempts")
        return False

    def clear_obstacles(self):
        """Removes all spawned obstacles"""
        if not self.obstacles:
            return
            
        if not self.delete_client.wait_for_service(timeout_sec=2.0):
             self.get_logger().warn('Delete service not available')
             return

        deleted_count = 0
        failed_deletes = []
        
        for name in self.obstacles:
            req = DeleteEntity.Request()
            req.name = name
            future = self.delete_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() and future.result().success:
                deleted_count += 1
                self.get_logger().info(f"Deleted obstacle: {name}")
            else:
                failed_deletes.append(name)
                self.get_logger().warn(f"Failed to delete {name}")
            time.sleep(0.05)  # Small delay between deletions
        
        self.obstacles = []
        self.get_logger().info(f"Cleared obstacles: {deleted_count} deleted, {len(failed_deletes)} failed")

    def spawn_dynamic_obstacle(self, name, x_range, y_range):
        """
        Placeholder for dynamic obstacle. 
        In strict sense, requires a plugin or external controller to move it.
        Here we just spawn a non-static model.
        """
        self.get_logger().info("Spawning dynamic obstacle (implementation pending)")
        # Similar to static, but set <static>false</static> and maybe add a simple plugin
        pass
