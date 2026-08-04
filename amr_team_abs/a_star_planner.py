#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import heapq
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import tf2_ros

# ---> FIX 1: Import the QoS policies so we can read the Map Server's bulletin board <---
from rclpy.qos import QoSProfile, DurabilityPolicy

def world_to_grid(x, y, info):
    res = info["resolution"]
    col = int((x - info["origin"][0]) / res)
    row = int((y - info["origin"][1]) / res)
    return row, col

def grid_to_world(row, col, info):
    res = info["resolution"]
    x = info["origin"][0] + (col + 0.5) * res
    y = info["origin"][1] + (row + 0.5) * res
    return x, y

def in_bounds(row, col, info):
    return 0 <= row < info["height"] and 0 <= col < info["width"]

def inflate(grid, info, robot_radius_m):
    blocked = (grid == 100) | (grid == -1)
    radius_cells = int(math.ceil(robot_radius_m / info["resolution"]))
    if radius_cells <= 0:
        return blocked
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~blocked)
    return dist <= radius_cells

NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]

def heuristic(a, b):
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

def astar(blocked, start, goal, info):
    if not in_bounds(*start, info) or not in_bounds(*goal, info):
        return None
    if blocked[start] or blocked[goal]:
        return None

    open_set = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current in closed: continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        r, c = current
        for dr, dc, step in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc, info) or blocked[nr, nc]: continue
            if dr != 0 and dc != 0 and blocked[r + dr, c] and blocked[r, c + dc]: continue

            tentative = g + step
            if tentative < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = tentative
                came_from[(nr, nc)] = current
                heapq.heappush(open_set, (tentative + heuristic((nr, nc), goal), tentative, (nr, nc)))
    return None

def simplify(path):
    if not path or len(path) < 3: return path
    out = [path[0]]
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        d2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        if d1 != d2: out.append(path[i])
    out.append(path[-1])
    return out


class AStarPlanner(Node):
    def __init__(self):
        super().__init__("a_star_planner")

        # ---> FIX 2: Apply the QoS Profile to the Map Subscriber <---
        qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_profile)

        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, "/global_path", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.info = None
        self.blocked_grid = None
        self.robot_radius = 0.35

        self.get_logger().info("A* Global Planner Ready. Waiting for Map...")

    def map_callback(self, msg):
        self.info = {
            "resolution": msg.info.resolution,
            "origin": [msg.info.origin.position.x, msg.info.origin.position.y],
            "height": msg.info.height,
            "width": msg.info.width,
        }
        grid_1d = np.array(msg.data, dtype=np.int8)
        grid_2d = grid_1d.reshape((self.info["height"], self.info["width"]))

        self.blocked_grid = inflate(grid_2d, self.info, self.robot_radius)
        self.get_logger().info("Map received and obstacles inflated!")

    def goal_callback(self, msg):
        if self.blocked_grid is None:
            self.get_logger().warn("Cannot plan: No map received yet.")
            return

        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            start_x = t.transform.translation.x
            start_y = t.transform.translation.y
        except Exception as e:
            self.get_logger().error(f"Could not find robot position: {e}")
            return

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        start_grid = world_to_grid(start_x, start_y, self.info)
        goal_grid = world_to_grid(goal_x, goal_y, self.info)

        self.get_logger().info(f"Planning from {start_grid} to {goal_grid}...")

        path_cells = astar(self.blocked_grid, start_grid, goal_grid, self.info)

        if path_cells is None:
            self.get_logger().error("NO PATH FOUND!")
            return

        waypoints = simplify(path_cells)

        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for r, c in waypoints:
            wx, wy = grid_to_world(r, c, self.info)
            pose = PoseStamped()

            # ---> FIX 3: Add the missing headers to the individual poses <---
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()

            pose.pose.position.x = wx
            pose.pose.position.y = wy
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published path with {len(waypoints)} waypoints!")


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
