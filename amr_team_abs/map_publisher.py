#!/usr/bin/env python3
"""Publish a ROS2 map (.pgm + .yaml) as a latched /map topic.

nav2 is not installed here, so map_server is unavailable. This does the
same job: read the map, publish one OccupancyGrid with TRANSIENT_LOCAL
QoS, stay alive so late subscribers still get it.

  ros2 run amr-team-abs map_publisher --ros-args -p map_yaml:=/path/to/map.yaml
"""

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid


class MapPublisher(Node):

    def __init__(self):
        super().__init__("map_publisher")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("topic", "/map")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("unknown_pixel", 205)
        # Republish periodically. TRANSIENT_LOCAL should deliver to late
        # subscribers on its own, but it is not reliable over the
        # FastRTPS TCP transport used on the Robile network. 0 disables.
        self.declare_parameter("republish_period", 2.0)

        yaml_path = self.get_parameter("map_yaml").value
        if not yaml_path:
            self.get_logger().error(
                "map_yaml parameter is empty. Pass "
                "--ros-args -p map_yaml:=/full/path/to/map.yaml")
            raise SystemExit(1)

        yaml_path = os.path.expanduser(yaml_path)
        if not os.path.exists(yaml_path):
            self.get_logger().error("Map yaml not found: %s" % yaml_path)
            raise SystemExit(1)

        try:
            grid, info = self.load(yaml_path)
        except FileNotFoundError as e:
            self.get_logger().error(
                "Could not open the map image: %s\n"
                "The 'image:' field in %s must name a file sitting next to "
                "it." % (e, yaml_path))
            raise SystemExit(1)

        self.msg = self.build_message(grid, info)

        qos = QoSProfile(depth=1,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(OccupancyGrid,
                                         self.get_parameter("topic").value, qos)
        self.publish()

        period = float(self.get_parameter("republish_period").value)
        if period > 0:
            self.create_timer(period, self.publish)

        w, h, r = info["width"], info["height"], info["resolution"]
        ox, oy = info["origin"]
        self.get_logger().info(
            "Published %dx%d @ %s m = %.2f x %.2f m | x %.2f..%.2f | "
            "y %.2f..%.2f | free %d occ %d unknown %d"
            % (w, h, r, w * r, h * r, ox, ox + w * r, oy, oy + h * r,
               int((grid == 0).sum()), int((grid == 100).sum()),
               int((grid == -1).sum())))

        if info["free_thresh"] > 0.196:
            self.get_logger().warn(
                "free_thresh is %.3f in the yaml. Unknown grey (pixel 205) "
                "gives occ = 0.196, which is below that, so a standard "
                "map_server would read all unmapped space as FREE. Set "
                "free_thresh: 0.196." % info["free_thresh"])

    def build_message(self, grid, info):
        msg = OccupancyGrid()
        msg.header.frame_id = self.get_parameter("frame_id").value
        msg.info.resolution = info["resolution"]
        msg.info.width = info["width"]
        msg.info.height = info["height"]
        msg.info.origin.position.x = info["origin"][0]
        msg.info.origin.position.y = info["origin"][1]
        yaw = info["origin_yaw"]
        msg.info.origin.orientation.z = math.sin(yaw * 0.5)
        msg.info.origin.orientation.w = math.cos(yaw * 0.5)
        # OccupancyGrid.data is row-major with row 0 at the BOTTOM,
        # which is how `grid` is stored after the flip in load().
        msg.data = grid.ravel().astype(np.int8).tolist()
        return msg

    def publish(self):
        # Re-stamp every time. In sim the node may construct before
        # /clock exists, which would leave a stale or zero stamp.
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.info.map_load_time = self.msg.header.stamp
        self.pub.publish(self.msg)

    def load(self, yaml_path):
        import yaml
        from PIL import Image

        with open(yaml_path) as f:
            meta = yaml.safe_load(f)

        img_path = meta["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)),
                                    img_path)

        raw = np.array(Image.open(img_path).convert("L"))
        # .pgm row 0 is the TOP; the map frame origin is BOTTOM-left.
        raw = np.flipud(raw)

        p = raw / 255.0 if int(meta.get("negate", 0)) else (255.0 - raw) / 255.0

        grid = np.full(p.shape, -1, dtype=np.int8)
        grid[p > float(meta.get("occupied_thresh", 0.65))] = 100
        grid[p < float(meta.get("free_thresh", 0.25))] = 0

        unknown_pixel = int(self.get_parameter("unknown_pixel").value)
        if unknown_pixel >= 0:
            n_forced = int(((raw == unknown_pixel) & (grid != -1)).sum())
            grid[raw == unknown_pixel] = -1
            if n_forced:
                self.get_logger().warn(
                    "%d pixels of value %d forced to unknown."
                    % (n_forced, unknown_pixel))

        origin = [float(v) for v in meta["origin"]]
        origin_yaw = origin[2] if len(origin) > 2 else 0.0
        if abs(origin_yaw) > 1e-6:
            self.get_logger().warn(
                "Map origin has a yaw of %.3f rad. Our A* and particle "
                "filter both assume an axis-aligned origin, so coordinates "
                "will be wrong. Re-save the map with origin yaw 0."
                % origin_yaw)

        info = {
            "resolution": float(meta["resolution"]),
            "origin": (origin[0], origin[1]),
            "origin_yaw": origin_yaw,
            "width": int(grid.shape[1]),
            "height": int(grid.shape[0]),
            "free_thresh": float(meta.get("free_thresh", 0.25)),
        }
        return grid, info


def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
