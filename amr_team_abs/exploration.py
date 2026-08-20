#!/usr/bin/env python3

import rclpy
import tf2_ros
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


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

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.map_received = False

        self.get_logger().info(
            "Exploration node started. Waiting for map..."
        )

    def map_callback(self, msg):
        self.map_received = True

        width = msg.info.width
        height = msg.info.height
        grid = list(msg.data)

        frontiers = find_frontiers(
            grid,
            width,
            height
        )

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

        self.get_logger().info(
            f"Robot position: x={robot_x:.2f}, y={robot_y:.2f}"
        )

        if frontiers:
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

            self.get_logger().info(
                f"Closest frontier: "
                f"x={closest_frontier[0]:.2f}, "
                f"y={closest_frontier[1]:.2f}, "
                f"distance={closest_distance:.2f} m"
            )

        self.get_logger().info(
            f"Map received: {width} x {height}, "
            f"resolution={msg.info.resolution}"
        )

        self.get_logger().info(
            f"Frontier cells detected: {len(frontiers)}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = ExplorationNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
