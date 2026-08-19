#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


def find_frontiers(grid, width, height):
    frontiers = []

    for row in range(1, height - 1):
        for col in range(1, width - 1):

            index = row * width + col

            # Frontier must be a free cell
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

                # Unknown cell next to free cell = frontier
                if grid[neighbour_index] == -1:
                    frontiers.append((row, col))
                    break

    return frontiers


class ExplorationNode(Node):

    def __init__(self):
        super().__init__("exploration_node")

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            10
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
