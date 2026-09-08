#!/usr/bin/env python3

import math
import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from rclpy.qos import QoSProfile, DurabilityPolicy

class ExplorationMapperNode(Node):
    def __init__(self):
        super().__init__("exploration_mapper")

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.laser_offset_x = None
        self.laser_offset_y = None
        self.latest_sensor_stamp = None

        # Map parameters
        self.map_size = 20.0
        self.resolution = 0.1
        self.width = int(self.map_size / self.resolution)
        self.height = int(self.map_size / self.resolution)

        # map origin at (-10,-10)
        self.origin_x = -10.0
        self.origin_y = -10.0

        # Assignment parameters
        self.p_occ = 0.1
        self.p_free = 0.8
        self.min_log = -2.0
        self.max_log = 2.0
        self.l_occ = math.log(self.p_occ / (1.0 - self.p_occ))
        self.l_free = math.log(self.p_free / (1.0 - self.p_free))

        self.last_update_x = None
        self.last_update_y = None
        self.last_update_yaw = None
        self.translation_threshold = 0.05
        self.rotation_threshold = 0.05

        # log-odds map
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

        # Exploration State Machine
        self.active_goal_world = None
        self.active_goal_grid = None
        self.goal_start_time = None
        self.goal_start_pose = None

        self.blacklist = set()
        self.goal_tolerance = 0.3
        self.timeout_seconds = 15.0
        self.stuck_distance = 0.2

        # Initialization Spin
        self.is_spinning = True
        self.spin_start_time = None
        self.spin_duration = 8.0
        self.spin_speed = 0.45

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("Unified SLAM & Exploration Node Started!")

    def update_cell(self, gx, gy, value):
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return
        self.log_odds[gy, gx] += value
        self.log_odds[gy, gx] = np.clip(
            self.log_odds[gy, gx], self.min_log, self.max_log
        )
    # def update_cell(self, gx, gy, value):
    #     if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
    #         return

    #     # ANTI-ERASURE LOCK: Prevent odometry drift from deleting confirmed walls.
    #     # If the cell is strongly occupied (log_odds < -2.0) and the incoming
    #     # ray tries to mark it as free space (value > 0), reject the update.
    #     if value > 0 and self.log_odds[gy, gx] < -2.0:
    #         return

    #     self.log_odds[gy, gx] += value
    #     self.log_odds[gy, gx] = np.clip(
    #         self.log_odds[gy, gx], self.min_log, self.max_log
    #     )

    def odom_callback(self, msg):
        self.latest_sensor_stamp = msg.header.stamp
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        self.latest_sensor_stamp = msg.header.stamp

        # Pause mapping during the initial spin
        if self.is_spinning:
            return

        # Only integrate a scan after meaningful motion. Mapping every
        # scan at 10-40 Hz bakes the same (drifting) odometry pose into
        # the grid over and over, which is what makes walls smear.
        if self.last_update_x is not None:
            moved = math.hypot(self.robot_x - self.last_update_x,
                               self.robot_y - self.last_update_y)
            dyaw = self.robot_yaw - self.last_update_yaw
            dyaw = (dyaw + math.pi) % (2.0 * math.pi) - math.pi
            if moved < self.translation_threshold and abs(dyaw) < self.rotation_threshold:
                return
        self.last_update_x = self.robot_x
        self.last_update_y = self.robot_y
        self.last_update_yaw = self.robot_yaw

        try:
            t = self.tf_buffer.lookup_transform(
                "odom",
                msg.header.frame_id,
                msg.header.stamp,
                rclpy.duration.Duration(seconds=0.05)
            )
            laser_world_x = t.transform.translation.x
            laser_world_y = t.transform.translation.y
            q = t.transform.rotation
            laser_world_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        except Exception:
            return

        gx_laser, gy_laser = self.world_to_grid(laser_world_x, laser_world_y)

        free_cells = set()
        occ_cells = set()

        for beam_idx, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r) or r < msg.range_min or r > msg.range_max:
                continue

            if r < 0.5:
                continue

            is_max_range = (r >= msg.range_max - 0.1) or (r >= 8.0)

            angle = msg.angle_min + beam_idx * msg.angle_increment
            world_angle = laser_world_yaw + angle
            x_hit = laser_world_x + r * math.cos(world_angle)
            y_hit = laser_world_y + r * math.sin(world_angle)
            gx_hit, gy_hit = self.world_to_grid(x_hit, y_hit)

            ray_cells = self.bresenham(gx_laser, gy_laser, gx_hit, gy_hit)
            if not ray_cells:
                continue

            for gx, gy in ray_cells[:-1]:
                free_cells.add((gx, gy))

            if not is_max_range:
                gx_occ, gy_occ = ray_cells[-1]
                occ_cells.add((gx_occ, gy_occ))
            else:
                free_cells.add(ray_cells[-1])

        free_cells -= occ_cells

        for gx, gy in free_cells:
            self.update_cell(gx, gy, self.l_free)
        for gx, gy in occ_cells:
            self.update_cell(gx, gy, self.l_occ)

    def timer_callback(self):
        if self.latest_sensor_stamp is None:
            return

        t = TransformStamped()
        t.header.stamp = self.latest_sensor_stamp
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        p = 1.0 / (1.0 + np.exp(-self.log_odds))
        map_data = np.full(self.log_odds.shape, -1, dtype=np.int8)
        map_data[p > 0.7] = 0
        map_data[p < 0.4] = 100

        msg = OccupancyGrid()
        msg.header.stamp = self.latest_sensor_stamp
        msg.header.frame_id = "map"
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = map_data.flatten().tolist()
        self.map_pub.publish(msg)

        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.is_spinning:
            if self.spin_start_time is None:
                self.spin_start_time = current_time

            if (current_time - self.spin_start_time) < self.spin_duration:
                spin_cmd = Twist()
                spin_cmd.angular.z = self.spin_speed
                self.cmd_pub.publish(spin_cmd)
                self.get_logger().info("Spinning to clear physical space. MAPPING PAUSED...")
                return
            else:
                self.is_spinning = False
                self.cmd_pub.publish(Twist())
                self.get_logger().info("Spin complete! Commencing mapping and exploration.")

        self.explore(map_data)

    def explore(self, grid_2d):
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.active_goal_world is not None:
            gx, gy = self.active_goal_world
            dist_to_goal = math.hypot(gx - self.robot_x, gy - self.robot_y)

            if dist_to_goal < self.goal_tolerance:
                self.get_logger().info("Frontier Reached! Scanning for next target...")
                self.active_goal_world = None
                return

            time_elapsed = current_time - self.goal_start_time
            if time_elapsed > self.timeout_seconds:
                dist_moved = math.hypot(self.robot_x - self.goal_start_pose[0], self.robot_y - self.goal_start_pose[1])

                if dist_moved < self.stuck_distance:
                    self.get_logger().warn("A* FAILED OR ROBOT STUCK! Blacklisting this frontier.")
                    self.blacklist.add(self.active_goal_grid)
                    self.active_goal_world = None
                else:
                    self.goal_start_time = current_time
                    self.goal_start_pose = (self.robot_x, self.robot_y)
            return

        free_space = (grid_2d == 0)
        unknown_space = (grid_2d == -1)

        up = np.roll(unknown_space, 1, axis=0)
        down = np.roll(unknown_space, -1, axis=0)
        left = np.roll(unknown_space, 1, axis=1)
        right = np.roll(unknown_space, -1, axis=1)

        frontiers_mask = free_space & (up | down | left | right)
        frontier_rows, frontier_cols = np.where(frontiers_mask)

        if len(frontier_rows) == 0:
            self.get_logger().info("NO FRONTIERS FOUND! EXPLORATION COMPLETE!")
            return

        closest_dist = float("inf")
        best_frontier_grid = None
        best_frontier_world = None

        for row, col in zip(frontier_rows, frontier_cols):
            if (row, col) in self.blacklist:
                continue

            fx, fy = self.grid_to_world(col, row)
            dist = math.hypot(fx - self.robot_x, fy - self.robot_y)

            # FIX: Ignore any frontier closer than 0.6m.
            # This ensures the post-pull-back target stays outside the 0.3m goal_tolerance.
            if dist <= 0.6:
                self.blacklist.add((row, col))
                continue

            if dist < closest_dist:
                closest_dist = dist
                best_frontier_grid = (row, col)
                best_frontier_world = (fx, fy)

        if best_frontier_world:
            fx, fy = best_frontier_world
            dx = fx - self.robot_x
            dy = fy - self.robot_y
            dist_to_frontier = math.hypot(dx, dy)

            # Universal pull-back: Always pull back by 0.4 meters (or halfway if the frontier is very close)
            # This guarantees the goal never lands on the raw frontier edge inside A*'s inflation zone.
            pull_back_dist = min(0.4, dist_to_frontier * 0.5)
            scale = (dist_to_frontier - pull_back_dist) / max(dist_to_frontier, 1e-5)

            target_x = self.robot_x + dx * scale
            target_y = self.robot_y + dy * scale

            self.active_goal_grid = best_frontier_grid
            self.active_goal_world = (target_x, target_y)
            self.goal_start_time = current_time
            self.goal_start_pose = (self.robot_x, self.robot_y)

            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.latest_sensor_stamp
            goal_msg.header.frame_id = "map"
            goal_msg.pose.position.x = target_x
            goal_msg.pose.position.y = target_y
            goal_msg.pose.orientation.w = 1.0

            self.goal_pub.publish(goal_msg)
            self.get_logger().info(f"Published Safe Shifted Goal: x={target_x:.2f}, y={target_y:.2f}")

    def world_to_grid(self, x, y):
        gx = math.floor((x - self.origin_x) / self.resolution)
        gy = math.floor((y - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
            x = self.origin_x + (gx + 0.5) * self.resolution
            y = self.origin_y + (gy + 0.5) * self.resolution
            return x, y

    def bresenham(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

def main(args=None):
    rclpy.init(args=args)
    node = ExplorationMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
