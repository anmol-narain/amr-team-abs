#!/usr/bin/env python3
"""Headless robot simulator that raycasts against a real map.

my_lab_map5new is a SLAM map of the lab - there is no Gazebo world that
matches it. This fakes the robot instead: holds a ground-truth pose,
drives it from /cmd_vel, raycasts the map to produce /scan, publishes
odometry with drift. To the particle filter it looks like a real robot.

It knows the true pose, so it prints the real localisation error every
second - a number Gazebo will not give you for free.

  publishes: /scan, /odom, TF odom->base_link, TF base_link->laser,
             /ground_truth (PoseStamped)
  subscribes: /cmd_vel, /mcl_pose
"""

import math
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import (Twist, TransformStamped, PoseStamped,
                               PoseWithCovarianceStamped)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import tf2_ros


def yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


def quat_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def normalize_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_map(yaml_path):
    """Same convention as everything else: row 0 = bottom."""
    import yaml
    from PIL import Image

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img_path = meta["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)),
                                img_path)

    raw = np.flipud(np.array(Image.open(img_path).convert("L")))
    p = raw / 255.0 if int(meta.get("negate", 0)) else (255.0 - raw) / 255.0

    occupied = p > float(meta.get("occupied_thresh", 0.65))
    # Unmapped space blocks the laser too - a real lidar sees the walls
    # of the room it is in, not through them.
    unknown = (raw == 205)
    blocking = occupied | unknown

    return blocking, {
        "resolution": float(meta["resolution"]),
        "origin": (float(meta["origin"][0]), float(meta["origin"][1])),
        "width": int(raw.shape[1]),
        "height": int(raw.shape[0]),
    }


def raycast(blocking, info, x, y, yaw, angles, max_range, step):
    """Vectorised ray march: all beams forward together, first hit wins."""
    res = info["resolution"]
    ox, oy = info["origin"]
    h, w = blocking.shape

    d = np.arange(step, max_range + step, step)
    a = yaw + angles
    px = x + np.cos(a)[:, None] * d[None, :]
    py = y + np.sin(a)[:, None] * d[None, :]

    col = ((px - ox) / res).astype(np.int32)
    row = ((py - oy) / res).astype(np.int32)
    outside = (col < 0) | (col >= w) | (row < 0) | (row >= h)
    np.clip(col, 0, w - 1, out=col)
    np.clip(row, 0, h - 1, out=row)

    hit = blocking[row, col] | outside
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1)

    return np.where(any_hit, d[first], float("inf"))


class FakeRobot(Node):

    def __init__(self):
        super().__init__("fake_robot_sim")
        p = self.declare_parameter

        p("map_yaml", "")
        p("start_x", -1.52)
        p("start_y", -7.15)
        p("start_yaw", 0.0)

        p("odom_frame", "odom")
        p("base_frame", "base_link")
        p("laser_frame", "base_laser_front_link")
        p("laser_x", 0.25)
        p("laser_y", 0.0)

        p("num_beams", 360)
        p("angle_min", -math.pi)
        p("angle_max", math.pi)
        p("range_min", 0.10)
        p("range_max", 8.0)
        p("scan_rate", 10.0)
        p("range_noise", 0.02)
        p("raycast_step", 0.02)

        p("odom_drift", 0.02)
        p("max_linear", 0.6)
        p("max_angular", 1.5)

        g = lambda k: self.get_parameter(k).value

        yaml_path = g("map_yaml")
        if not yaml_path:
            self.get_logger().error("map_yaml parameter is empty.")
            raise SystemExit(1)
        self.blocking, self.info = load_map(yaml_path)

        self.odom_frame = g("odom_frame")
        self.base_frame = g("base_frame")
        self.laser_frame = g("laser_frame")
        self.laser_x = float(g("laser_x"))
        self.laser_y = float(g("laser_y"))

        self.n_beams = int(g("num_beams"))
        self.angle_min = float(g("angle_min"))
        self.angle_max = float(g("angle_max"))
        self.range_min = float(g("range_min"))
        self.range_max = float(g("range_max"))
        self.range_noise = float(g("range_noise"))
        self.raycast_step = float(g("raycast_step"))
        self.drift = float(g("odom_drift"))
        self.max_linear = float(g("max_linear"))
        self.max_angular = float(g("max_angular"))

        self.angles = np.linspace(self.angle_min, self.angle_max,
                                  self.n_beams, endpoint=False)
        self.angle_increment = (self.angle_max - self.angle_min) / self.n_beams

        self.tx = float(g("start_x"))
        self.ty = float(g("start_y"))
        self.tyaw = float(g("start_yaw"))

        self.ox_ = 0.0
        self.oy_ = 0.0
        self.oyaw_ = 0.0

        self.cmd = (0.0, 0.0, 0.0)
        self.rng = np.random.default_rng()
        self.mcl_pose = None

        res = self.info["resolution"]
        gx = int((self.tx - self.info["origin"][0]) / res)
        gy = int((self.ty - self.info["origin"][1]) / res)
        if not (0 <= gx < self.info["width"] and 0 <= gy < self.info["height"]) \
                or self.blocking[gy, gx]:
            self.get_logger().error(
                "Start pose (%.2f, %.2f) is inside a wall or off the map. "
                "Map covers x %.2f..%.2f, y %.2f..%.2f."
                % (self.tx, self.ty,
                   self.info["origin"][0],
                   self.info["origin"][0] + self.info["width"] * res,
                   self.info["origin"][1],
                   self.info["origin"][1] + self.info["height"] * res))
            raise SystemExit(1)

        sensor_qos = QoSProfile(depth=5,
                                reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST)

        self.scan_pub = self.create_publisher(LaserScan, "/scan", sensor_qos)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.truth_pub = self.create_publisher(PoseStamped, "/ground_truth", 10)

        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/mcl_pose",
                                 self.mcl_callback, 10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.static_tf = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_laser_tf()

        rate = float(g("scan_rate"))
        self.dt = 1.0 / rate
        self.create_timer(self.dt, self.tick)
        self.create_timer(1.0, self.report)

        self.get_logger().info(
            "Fake robot on %s. Truth starts at (%.2f, %.2f, %.0f deg). "
            "%d beams, drift %.0f%%. Drive it with teleop_twist_keyboard."
            % (os.path.basename(yaml_path), self.tx, self.ty,
               math.degrees(self.tyaw), self.n_beams, self.drift * 100))

    def publish_laser_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_frame
        t.child_frame_id = self.laser_frame
        t.transform.translation.x = self.laser_x
        t.transform.translation.y = self.laser_y
        t.transform.rotation.w = 1.0
        self.static_tf.sendTransform(t)

    def cmd_callback(self, msg):
        vx = max(-self.max_linear, min(self.max_linear, msg.linear.x))
        vy = max(-self.max_linear, min(self.max_linear, msg.linear.y))
        wz = max(-self.max_angular, min(self.max_angular, msg.angular.z))
        self.cmd = (vx, vy, wz)

    def mcl_callback(self, msg):
        q = msg.pose.pose.orientation
        self.mcl_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                         quat_to_yaw(q.x, q.y, q.z, q.w))

    def tick(self):
        vx, vy, wz = self.cmd
        dt = self.dt

        dx = (math.cos(self.tyaw) * vx - math.sin(self.tyaw) * vy) * dt
        dy = (math.sin(self.tyaw) * vx + math.cos(self.tyaw) * vy) * dt
        nx, ny = self.tx + dx, self.ty + dy

        res = self.info["resolution"]
        gx = int((nx - self.info["origin"][0]) / res)
        gy = int((ny - self.info["origin"][1]) / res)
        blocked = not (0 <= gx < self.info["width"]
                       and 0 <= gy < self.info["height"]) \
            or self.blocking[gy, gx]

        if not blocked:
            self.tx, self.ty = nx, ny
        self.tyaw = normalize_angle(self.tyaw + wz * dt)

        k = 1.0 + self.rng.normal(0.0, self.drift) if self.drift > 0 else 1.0
        kr = 1.0 + self.rng.normal(0.0, self.drift) if self.drift > 0 else 1.0
        odx = (math.cos(self.oyaw_) * vx - math.sin(self.oyaw_) * vy) * dt * k
        ody = (math.sin(self.oyaw_) * vx + math.cos(self.oyaw_) * vy) * dt * k
        if not blocked:
            self.ox_ += odx
            self.oy_ += ody
        self.oyaw_ = normalize_angle(self.oyaw_ + wz * dt * kr)

        stamp = self.get_clock().now().to_msg()
        self.publish_odom(stamp)
        self.publish_scan(stamp)
        self.publish_truth(stamp)

    def publish_odom(self, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = self.ox_
        t.transform.translation.y = self.oy_
        q = yaw_to_quat(self.oyaw_)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = self.ox_
        msg.pose.pose.position.y = self.oy_
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        msg.twist.twist.linear.x = self.cmd[0]
        msg.twist.twist.linear.y = self.cmd[1]
        msg.twist.twist.angular.z = self.cmd[2]
        self.odom_pub.publish(msg)

    def publish_scan(self, stamp):
        lx = self.tx + math.cos(self.tyaw) * self.laser_x \
            - math.sin(self.tyaw) * self.laser_y
        ly = self.ty + math.sin(self.tyaw) * self.laser_x \
            + math.cos(self.tyaw) * self.laser_y

        ranges = raycast(self.blocking, self.info, lx, ly, self.tyaw,
                         self.angles, self.range_max, self.raycast_step)

        finite = np.isfinite(ranges)
        if self.range_noise > 0:
            ranges = np.where(
                finite,
                ranges + self.rng.normal(0.0, self.range_noise, ranges.shape),
                ranges)
        ranges = np.where(finite, np.maximum(ranges, self.range_min), ranges)

        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = self.laser_frame
        msg.angle_min = float(self.angle_min)
        msg.angle_max = float(self.angle_max)
        msg.angle_increment = float(self.angle_increment)
        msg.time_increment = 0.0
        msg.scan_time = float(self.dt)
        msg.range_min = float(self.range_min)
        msg.range_max = float(self.range_max)
        msg.ranges = [float(r) for r in ranges]
        self.scan_pub.publish(msg)

    def publish_truth(self, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.pose.position.x = self.tx
        msg.pose.position.y = self.ty
        q = yaw_to_quat(self.tyaw)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.truth_pub.publish(msg)

    def report(self):
        if self.mcl_pose is None:
            return
        ex = self.mcl_pose[0] - self.tx
        ey = self.mcl_pose[1] - self.ty
        eyaw = normalize_angle(self.mcl_pose[2] - self.tyaw)
        self.get_logger().info(
            "truth (%+.2f, %+.2f, %+.0f)  mcl (%+.2f, %+.2f, %+.0f)  "
            "err %.3f m / %.1f deg"
            % (self.tx, self.ty, math.degrees(self.tyaw),
               self.mcl_pose[0], self.mcl_pose[1], math.degrees(self.mcl_pose[2]),
               math.hypot(ex, ey), abs(math.degrees(eyaw))))


def main(args=None):
    rclpy.init(args=args)
    node = FakeRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
