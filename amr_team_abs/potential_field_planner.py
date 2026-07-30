#!/usr/bin/env python3

import math
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion

class PotentialFieldPlanner(Node):
    def __init__(self):
        super().__init__("potential_field_planner")

        # Subscriber
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Static goal pose for Rosbag testing
        self.goal_x = 4.0
        self.goal_y = 1.0
        self.goal_theta = -1.0

        # Potential field parameters
        self.k_att = 1.0
        self.k_rep = 0.25
        self.rho_0 = 1.0

        # Velocity limits
        self.max_linear = 0.3
        self.max_angular = 0.3

        # Goal thresholds
        self.goal_threshold = 0.3
        self.angle_threshold = 0.1

        # Low-Pass Filter State Variables
        self.prev_v = 0.0
        self.prev_w = 0.0
        self.alpha = 0.3  # Smoothing factor (lower = smoother but slower to react)

        self.get_logger().info("Potential Field Planner Started (with Low-Pass Filter)")

    def scan_callback(self, scan_msg):

        # We try to get the transform at the time of the scan.
        # If it fails (common in rosbags), we fall back to the newest available transform.
        # Fail-safe live TF lookup: Grab the absolute latest available transform
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", "base_link", rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF Lookup Failed: {str(e)}")
            # Safety stop on real hardware if transform drops!
            stop_cmd = Twist()
            self.cmd_pub.publish(stop_cmd)
            return

        # Current robot position
        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y

        # Current robot orientation
        q = transform.transform.rotation
        quaternion = [q.x, q.y, q.z, q.w]
        _, _, robot_yaw = euler_from_quaternion(quaternion)

        # ==================================================
        # Attractive force toward goal
        # ==================================================
        dx = self.goal_x - robot_x
        dy = self.goal_y - robot_y
        dist_to_goal = math.sqrt(dx**2 + dy**2)

        att_x = self.k_att * dx
        att_y = self.k_att * dy

        # Normalize attractive vector (Conic potential)
        norm_att = math.sqrt(att_x**2 + att_y**2)
        if norm_att > 0:
            att_x /= norm_att
            att_y /= norm_att

        # ==================================================
        # Repulsive force from obstacles
        # ==================================================
        rep_x = 0.0
        rep_y = 0.0

        # Base angle rotated to global frame
        angle = robot_yaw + scan_msg.angle_min

        for r in scan_msg.ranges:
            if np.isinf(r) or np.isnan(r):
                angle += scan_msg.angle_increment
                continue

            # Only consider close obstacles
            if 0.05 < r < self.rho_0:
                # Obstacle vector in global frame
                obs_x = r * math.cos(angle)
                obs_y = r * math.sin(angle)

                # Repulsive magnitude
                repulsive_mag = self.k_rep * ((1.0 / r) - (1.0 / self.rho_0)) / (r**2)

                # Repulsive direction
                rep_x += -repulsive_mag * (obs_x / r)
                rep_y += -repulsive_mag * (obs_y / r)

            angle += scan_msg.angle_increment

        # ==================================================
        # Combine attractive and repulsive forces
        # ==================================================
        total_x = att_x + rep_x
        total_y = att_y + rep_y

        desired_heading = math.atan2(total_y, total_x)
        heading_error = self.normalize_angle(desired_heading - robot_yaw)

        # ==================================================
        # Motion control (Calculated first, then filtered)
        # ==================================================
        cmd = Twist()
        calculated_v = 0.0
        calculated_w = 0.0

        # Goal position reached
        if dist_to_goal < self.goal_threshold:
            final_angle_error = self.normalize_angle(self.goal_theta - robot_yaw)

            # Align final orientation
            if abs(final_angle_error) > self.angle_threshold:
                calculated_w = max(-self.max_angular, min(self.max_angular, 1.0 * final_angle_error))
                calculated_v = 0.0
            else:
                calculated_v = 0.0
                calculated_w = 0.0
                self.get_logger().info("Goal reached!")

        else:
            # Standard movement
            calculated_w = max(-self.max_angular, min(self.max_angular, 1.5 * heading_error))
            calculated_v = min(self.max_linear, 0.3 * dist_to_goal)

            # Slow down during sharp turns
            if abs(heading_error) > 0.5:
                calculated_v = 0.05

        # Apply the Low-Pass Filter
        cmd.linear.x = (self.alpha * calculated_v) + ((1.0 - self.alpha) * self.prev_v)
        cmd.angular.z = (self.alpha * calculated_w) + ((1.0 - self.alpha) * self.prev_w)

        # Save for the next loop
        self.prev_v = cmd.linear.x
        self.prev_w = cmd.angular.z

        # Publish velocity command
        self.cmd_pub.publish(cmd)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
