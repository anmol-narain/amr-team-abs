#!/usr/bin/env python3

import rclpy
import rclpy.duration
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy

def find_frontiers(grid, width, height):
    frontiers = []

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            index = row * width + col

            if grid[index] != 0:
                continue

            neighbours = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1)
            ]

            for neighbour_row, neighbour_col in neighbours:
                neighbour_index = (
                    neighbour_row * width + neighbour_col
                )

                if grid[neighbour_index] == -1:
                    frontiers.append((row, col))
                    break

    return frontiers


def grid_to_world(row, col, info):
    resolution = info.resolution
    origin_x = info.origin.position.x
    origin_y = info.origin.position.y

    x = origin_x + (col + 0.5) * resolution
    y = origin_y + (row + 0.5) * resolution

    return x, y


class ExplorationNode(Node):

    def __init__(self):
        super().__init__("exploration_node")
        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            qos_profile
        )
        
        self.goal_pub = self.create_publisher(
            PoseStamped,
            "/goal_pose",
            10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.map_received = False
        self.latest_map = None
        self.processed_map = False

        self.timer = self.create_timer(
            1.0,
            self.process_map
        )

        self.get_logger().info(
            "Exploration node started. Waiting for map..."
        )

    def map_callback(self, msg):
        self.map_received = True

        self.latest_map = msg
        self.processed_map = False

        self.get_logger().info(
            "Map received and stored. Waiting for TF..."
      )

    def process_map(self):
        if not self.map_received or self.latest_map is None:
            return

        if self.processed_map:
            return

        if not self.tf_buffer.can_transform(
            "map",
            "base_link",
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=2.0)
        ):
            self.get_logger().warn(
                "TF not ready yet. Waiting..."
            )
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )

            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y

        except Exception as e:
            self.get_logger().warn(
                f"Could not get robot position: {e}"
            )
            return

        msg = self.latest_map

        width = msg.info.width
        height = msg.info.height
        grid = list(msg.data)

        frontiers = find_frontiers(
            grid,
            width,
            height
        )

        self.get_logger().info(
            f"Robot position: x={robot_x:.2f}, y={robot_y:.2f}"
        )

        if not frontiers:
            self.get_logger().warn(
                "No frontiers found."
            )
            self.processed_map = True
            return

        closest_frontier = None
        closest_distance = float("inf")

        for row, col in frontiers:
            frontier_x, frontier_y = grid_to_world(
                row,
                col,
                msg.info
            )

            distance = (
                (frontier_x - robot_x) ** 2
                + (frontier_y - robot_y) ** 2
            ) ** 0.5

            if distance < closest_distance:
                closest_distance = distance
                closest_frontier = (
                    frontier_x,
                    frontier_y
                )

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()

        goal.pose.position.x = closest_frontier[0]
        goal.pose.position.y = closest_frontier[1]
        goal.pose.position.z = 0.0

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)

        self.get_logger().info(
            f"Goal published: x={closest_frontier[0]:.2f}, "
            f"y={closest_frontier[1]:.2f}"
        )

        self.processed_map = True

def main(args=None):
    rclpy.init(args=args)

    node = ExplorationNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
