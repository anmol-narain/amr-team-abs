#!/usr/bin/env python3

import math
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion

class PotentialFieldPlanner(Node):
    def __init__(self):
        super().__init__("potential_field_planner")

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.path_sub = self.create_subscription(Path, "/global_path", self.path_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Waypoint Manager State
        self.waypoints = []
        self.current_goal_idx = 0

        # Potential field parameters
        self.k_att = 1.0
        self.k_rep = 0.25
        self.rho_0 = 1.0

        # Velocity limits
        self.max_linear = 0.3  # Reduced slightly for physical safety
        self.max_angular = 0.3

        self.goal_threshold = 0.2

        # Low-Pass Filter parameters
        self.prev_v = 0.0
        self.prev_w = 0.0
        self.alpha = 0.6

        self.get_logger().info("Potential Field Planner Started (Waiting for A* Path)")

    def path_callback(self, msg):
        self.waypoints = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            self.waypoints.append((x, y))

        if self.waypoints:
            self.current_goal_idx = 0
            self.get_logger().info(f"Received new path with {len(self.waypoints)} waypoints!")

    def scan_callback(self, scan_msg):
        # Stop if there is no path yet
        if not self.waypoints or self.current_goal_idx >= len(self.waypoints):
            self.cmd_pub.publish(Twist())
            return

        try:
            # Live TF Lookup
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF Lookup Failed: {str(e)}")
            self.cmd_pub.publish(Twist())
            return

        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y

        q = transform.transform.rotation
        quaternion = [q.x, q.y, q.z, q.w]
        _, _, robot_yaw = euler_from_quaternion(quaternion)

        # ==================================================
        # The Glue: Waypoint Management
        # ==================================================
        # target_x, target_y = self.waypoints[self.current_goal_idx]
        # dx = target_x - robot_x
        # dy = target_y - robot_y
        # dist_to_target = math.sqrt(dx**2 + dy**2)

        # if dist_to_target < self.goal_threshold:
        #     self.current_goal_idx += 1
        #     if self.current_goal_idx >= len(self.waypoints):
        #         self.get_logger().info("Final Destination Reached!")
        #         self.cmd_pub.publish(Twist())
        #         self.waypoints = []
        #         return
        #     else:
        #         self.get_logger().info(f"Waypoint reached. Moving to waypoint {self.current_goal_idx}")
        #         target_x, target_y = self.waypoints[self.current_goal_idx]
        #         dx = target_x - robot_x
        #         dy = target_y - robot_y
        #         dist_to_target = math.sqrt(dx**2 + dy**2)
        #
        # ==================================================
        # The Glue: Smart Waypoint Management (Line-of-Sight)
        # ==================================================
        target_x, target_y = self.waypoints[self.current_goal_idx]
        dx = target_x - robot_x
        dy = target_y - robot_y
        dist_to_target = math.sqrt(dx**2 + dy**2)

        # Condition 1: We physically reached the waypoint
        if dist_to_target < self.goal_threshold:
            self.current_goal_idx += 1
            if self.current_goal_idx >= len(self.waypoints):
                self.get_logger().info("Final Destination Reached!")
                self.cmd_pub.publish(Twist())
                self.waypoints = []
                return
            else:
                self.get_logger().info(f"Waypoint reached. Moving to {self.current_goal_idx}")

        # Condition 2: Waypoint Starvation Recovery (Skip blocked waypoints)
        elif self.current_goal_idx + 1 < len(self.waypoints):
            # Check the distance to the NEXT waypoint
            next_x, next_y = self.waypoints[self.current_goal_idx + 1]
            next_dx = next_x - robot_x
            next_dy = next_y - robot_y
            dist_to_next = math.sqrt(next_dx**2 + next_dy**2)

            # If the next waypoint is closer than our current target,
            # we have likely been pushed past the current target by an obstacle. Skip it!
            if dist_to_next < dist_to_target:
                self.get_logger().info(f"Waypoint {self.current_goal_idx} blocked/Missed. Skipping to {self.current_goal_idx + 1}!")
                self.current_goal_idx += 1

        target_x, target_y = self.waypoints[self.current_goal_idx]
        dx = target_x - robot_x
        dy = target_y - robot_y
        dist_to_target = math.sqrt(dx**2 + dy**2)

        # ==================================================
        # Attractive force
        # ==================================================
        att_x = self.k_att * dx
        att_y = self.k_att * dy

        norm_att = math.sqrt(att_x**2 + att_y**2)
        if norm_att > 0:
            att_x /= norm_att
            att_y /= norm_att

        # ==================================================
        # Repulsive force
        # ==================================================
        rep_x = 0.0
        rep_y = 0.0
        angle = robot_yaw + scan_msg.angle_min

        for r in scan_msg.ranges:
            if np.isinf(r) or np.isnan(r):
                angle += scan_msg.angle_increment
                continue

            if 0.05 < r < self.rho_0:
                obs_x = r * math.cos(angle)
                obs_y = r * math.sin(angle)
                repulsive_mag = self.k_rep * ((1.0 / r) - (1.0 / self.rho_0)) / (r**2)
                rep_x += -repulsive_mag * (obs_x / r)
                rep_y += -repulsive_mag * (obs_y / r)

            angle += scan_msg.angle_increment

        total_x = att_x + rep_x
        total_y = att_y + rep_y

        desired_heading = math.atan2(total_y, total_x)
        heading_error = self.normalize_angle(desired_heading - robot_yaw)

        # ==================================================
        # Motion control with Low-Pass Filter
        # ==================================================
        cmd = Twist()
        calculated_w = max(-self.max_angular, min(self.max_angular, 1.5 * heading_error))
        calculated_v = min(self.max_linear, 0.3 * dist_to_target)

        if abs(heading_error) > 0.5:
            calculated_v = 0.05

        cmd.linear.x = (self.alpha * calculated_v) + ((1.0 - self.alpha) * self.prev_v)
        cmd.angular.z = (self.alpha * calculated_w) + ((1.0 - self.alpha) * self.prev_w)

        self.prev_v = cmd.linear.x
        self.prev_w = cmd.angular.z

        self.cmd_pub.publish(cmd)

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
