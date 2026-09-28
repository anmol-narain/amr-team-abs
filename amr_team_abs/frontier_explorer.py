#!/usr/bin/env python3
"""Frontier-based exploration.

This node does NOT build a map. slam_toolbox does that, and also
publishes the corrected map->odom transform. This node's job is the
part the coursework asks us to write: look at the map, find the
boundary between explored and unexplored space, choose a pose there,
and hand it to A* as a goal.

  in : /map              (nav_msgs/OccupancyGrid) from slam_toolbox
       TF map->base_link                          from slam_toolbox + drivers
  out: /goal_pose        (PoseStamped)            -> a_star_planner
       /frontier_markers (MarkerArray)            -> RViz
       /exploration_state(String)                 -> the current state

The state machine has five states:

  INIT_SPIN   rotate on the spot so slam_toolbox gets scans from every
              direction before we commit to a first goal
  SELECTING   detect frontier clusters, score them, publish a goal
  NAVIGATING  wait while A* and the potential field planner drive there
  RECOVERY    goal was unreachable or the robot got stuck: blacklist it
  DONE        no frontiers left

Selection strategies (`selection_strategy` parameter):
  nearest       closest cluster. Greedy, tends to zigzag.
  largest       biggest cluster. Maximises information, ignores travel.
  cost_utility  size_weight * normalised_size - distance. The usual
                compromise; tune size_weight to trade one against the
                other. This is the default.

Running two strategies over the same room and comparing coverage
against time is a cheap, honest experiment for the report.
"""

import math
from enum import Enum

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class State(Enum):
    INIT_SPIN = "INIT_SPIN"
    SELECTING = "SELECTING"
    NAVIGATING = "NAVIGATING"
    RECOVERY = "RECOVERY"
    DONE = "DONE"


# ======================================================================
#  Pure functions - no ROS, so they can be unit tested offline
# ======================================================================

def detect_frontier_clusters(grid, resolution, origin, min_cluster_cells,
                             robot_radius):
    """Find clusters of frontier cells and pick a safe pose in each.

    A frontier cell is FREE and has at least one UNKNOWN 4-neighbour -
    exactly the "boundary between the explored and unexplored region"
    the brief asks for.

    Raw frontier cells sit right on the edge of known space, often
    hugging a wall, where A*'s inflation would reject them as goals. So
    within each cluster we pick the cell with the greatest clearance
    from any obstacle, and drop clusters whose best cell is still
    inside the robot radius.
    """
    from scipy.ndimage import label, distance_transform_edt

    free = (grid == 0)
    unknown = (grid == -1)
    occupied = (grid == 100)

    # np.roll wraps around the array edges, which would invent frontiers
    # on the opposite side of the map. Clear the wrapped row/column.
    up = np.roll(unknown, 1, axis=0)
    up[0, :] = False
    down = np.roll(unknown, -1, axis=0)
    down[-1, :] = False
    left = np.roll(unknown, 1, axis=1)
    left[:, 0] = False
    right = np.roll(unknown, -1, axis=1)
    right[:, -1] = False

    frontier_mask = free & (up | down | left | right)
    if not frontier_mask.any():
        return []

    clearance = distance_transform_edt(~occupied) * resolution
    labels, n = label(frontier_mask, structure=np.ones((3, 3)))
    if n == 0:
        return []

    clusters = []
    for i in range(1, n + 1):
        cells = np.argwhere(labels == i)
        if cells.shape[0] < min_cluster_cells:
            continue                      # sensor noise, not a frontier

        cl = clearance[cells[:, 0], cells[:, 1]]
        best = int(np.argmax(cl))
        if cl[best] < robot_radius:
            continue                      # nowhere in here the robot fits

        r, c = cells[best]
        clusters.append({
            "size": int(cells.shape[0]),
            "clearance": float(cl[best]),
            "world": (origin[0] + (c + 0.5) * resolution,
                      origin[1] + (r + 0.5) * resolution),
        })
    return clusters


def score_clusters(clusters, robot_x, robot_y, strategy, size_weight):
    """Pick one cluster. Returns None if the list is empty."""
    if not clusters:
        return None

    for f in clusters:
        f["dist"] = math.hypot(f["world"][0] - robot_x,
                               f["world"][1] - robot_y)

    if strategy == "nearest":
        return min(clusters, key=lambda f: f["dist"])
    if strategy == "largest":
        return max(clusters, key=lambda f: f["size"])

    # cost_utility: reward big clusters, penalise travel
    max_size = max(f["size"] for f in clusters)
    for f in clusters:
        f["score"] = size_weight * (f["size"] / max_size) - f["dist"]
    return max(clusters, key=lambda f: f["score"])


# ======================================================================
#  ROS node
# ======================================================================

class FrontierExplorer(Node):

    def __init__(self):
        super().__init__("frontier_explorer")
        p = self.declare_parameter

        p("global_frame", "map")
        p("base_frame", "base_link")
        p("map_topic", "/map")
        p("goal_topic", "/goal_pose")

        # nearest | largest | cost_utility
        p("selection_strategy", "cost_utility")
        p("size_weight", 3.0)
        p("min_cluster_cells", 8)
        p("robot_radius", 0.25)
        p("min_goal_distance", 0.5)

        p("goal_republish", True)
        p("goal_republish_period", 5.0)
        p("goal_tolerance", 0.35)
        p("goal_timeout", 25.0)
        p("stuck_distance", 0.15)
        p("blacklist_radius", 0.4)
        p("max_failures", 3)

        p("initial_spin", True)
        p("spin_duration", 8.0)
        p("spin_speed", 0.4)

        p("planning_period", 1.0)

        g = lambda k: self.get_parameter(k).value
        self.global_frame = g("global_frame")
        self.base_frame = g("base_frame")
        self.strategy = g("selection_strategy")
        self.size_weight = float(g("size_weight"))
        self.min_cluster = int(g("min_cluster_cells"))
        self.robot_radius = float(g("robot_radius"))
        self.min_goal_distance = float(g("min_goal_distance"))
        self.goal_republish = bool(g("goal_republish"))
        self.goal_republish_period = float(g("goal_republish_period"))
        self.last_republish = 0.0
        self.goal_tolerance = float(g("goal_tolerance"))
        self.goal_timeout = float(g("goal_timeout"))
        self.stuck_distance = float(g("stuck_distance"))
        self.blacklist_radius = float(g("blacklist_radius"))
        self.max_failures = int(g("max_failures"))
        self.spin_duration = float(g("spin_duration"))
        self.spin_speed = float(g("spin_speed"))

        self.state = State.INIT_SPIN if g("initial_spin") else State.SELECTING
        self.spin_start = None

        self.grid = None
        self.map_info = None
        self.clusters = []

        self.goal = None            # (x, y) in the map frame
        self.goal_start_time = None
        self.goal_start_pose = None

        # Blacklist in WORLD coordinates, not grid cells. slam_toolbox
        # grows the map and shifts its origin, so a (row, col) recorded
        # now would refer to a different place later.
        self.blacklist = []
        self.failures = 0

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, g("map_topic"),
                                 self.map_callback, latched)

        self.goal_pub = self.create_publisher(PoseStamped, g("goal_topic"), 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.marker_pub = self.create_publisher(MarkerArray,
                                                "/frontier_markers", 10)
        self.state_pub = self.create_publisher(String, "/exploration_state", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_timer(float(g("planning_period")), self.tick)

        self.get_logger().info(
            "Frontier explorer up. strategy=%s. Mapping is slam_toolbox's "
            "job; this node only picks goals." % self.strategy)

    # ------------------------------------------------------------------

    def map_callback(self, msg):
        # slam_toolbox resizes the map and moves its origin as it goes,
        # so read the geometry fresh every time.
        self.map_info = {
            "resolution": msg.info.resolution,
            "origin": (msg.info.origin.position.x, msg.info.origin.position.y),
            "width": msg.info.width,
            "height": msg.info.height,
        }
        self.grid = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))

    def robot_pose(self):
        """Pose from TF, not from /odom.

        slam_toolbox publishes a real map->odom correction, so /odom
        alone is in the wrong frame. Asking TF for map->base_link gives
        the corrected pose and keeps this node consistent with A*, which
        does the same thing.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                self.global_frame, self.base_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn("map->base_link unavailable: %s" % e,
                                   throttle_duration_sec=5.0)
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.transform.translation.x, t.transform.translation.y, yaw)

    def is_blacklisted(self, x, y):
        for bx, by in self.blacklist:
            if math.hypot(x - bx, y - by) < self.blacklist_radius:
                return True
        return False

    def set_state(self, new_state, reason=""):
        if new_state != self.state:
            self.get_logger().info("%s -> %s%s"
                                   % (self.state.value, new_state.value,
                                      ("  (%s)" % reason) if reason else ""))
            self.state = new_state
        msg = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)

    # ------------------------------------------------------------------
    #  the state machine
    # ------------------------------------------------------------------

    def tick(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.state == State.INIT_SPIN:
            self.do_spin(now)
            return

        if self.state == State.DONE:
            self.set_state(State.DONE)
            return

        if self.grid is None:
            self.get_logger().warn("waiting for /map from slam_toolbox",
                                   throttle_duration_sec=5.0)
            return

        pose = self.robot_pose()
        if pose is None:
            return
        rx, ry, _ = pose

        if self.state == State.NAVIGATING:
            self.do_navigating(now, rx, ry)
        elif self.state == State.RECOVERY:
            self.do_recovery()
        elif self.state == State.SELECTING:
            self.do_selecting(now, rx, ry)

    def do_spin(self, now):
        """Rotate in place so slam_toolbox sees in every direction before
        we ask it where the frontiers are. Nothing else publishes
        /cmd_vel yet, because the launch file starts the planners after
        this finishes."""
        if self.spin_start is None:
            self.spin_start = now
            self.get_logger().info("Initial spin, %.0f s" % self.spin_duration)

        if now - self.spin_start < self.spin_duration:
            cmd = Twist()
            cmd.angular.z = self.spin_speed
            self.cmd_pub.publish(cmd)
            return

        self.cmd_pub.publish(Twist())
        self.set_state(State.SELECTING, "spin complete")

    def publish_goal(self, rx, ry):
        """Publish the current goal to A*.

        Called every tick while NAVIGATING, not just once. /goal_pose is
        not latched, so a planner that starts a moment later than us -
        which is exactly what the launch staging causes - would
        otherwise never see the first goal, and the robot would sit
        still until the stuck timeout fired.
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.global_frame
        msg.pose.position.x = float(self.goal[0])
        msg.pose.position.y = float(self.goal[1])
        # Face the way we are travelling, so the laser looks into the
        # unexplored space on arrival.
        yaw = math.atan2(self.goal[1] - ry, self.goal[0] - rx)
        msg.pose.orientation.z = math.sin(yaw * 0.5)
        msg.pose.orientation.w = math.cos(yaw * 0.5)
        self.goal_pub.publish(msg)

    def do_navigating(self, now, rx, ry):
        gx, gy = self.goal
        if math.hypot(gx - rx, gy - ry) < self.goal_tolerance:
            self.failures = 0
            self.goal = None
            self.set_state(State.SELECTING, "frontier reached")
            return

        if (self.goal_republish
                and now - self.last_republish > self.goal_republish_period):
            self.publish_goal(rx, ry)
            self.last_republish = now

        if now - self.goal_start_time > self.goal_timeout:
            moved = math.hypot(rx - self.goal_start_pose[0],
                               ry - self.goal_start_pose[1])
            if moved < self.stuck_distance:
                self.set_state(State.RECOVERY, "stuck, moved %.2f m" % moved)
            else:
                # Making progress, just slowly. Give it another window.
                self.goal_start_time = now
                self.goal_start_pose = (rx, ry)

    def do_recovery(self):
        if self.goal is not None:
            self.blacklist.append(self.goal)
            self.get_logger().warn(
                "Blacklisted (%.2f, %.2f). %d entries."
                % (self.goal[0], self.goal[1], len(self.blacklist)))
            self.goal = None

        self.failures += 1
        if self.failures >= self.max_failures:
            self.get_logger().error(
                "%d consecutive failures - stopping." % self.failures)
            self.cmd_pub.publish(Twist())
            self.set_state(State.DONE, "too many failures")
            return

        self.set_state(State.SELECTING, "retrying")

    def do_selecting(self, now, rx, ry):
        self.clusters = detect_frontier_clusters(
            self.grid, self.map_info["resolution"], self.map_info["origin"],
            self.min_cluster, self.robot_radius)

        candidates = [
            f for f in self.clusters
            if not self.is_blacklisted(*f["world"])
            and math.hypot(f["world"][0] - rx, f["world"][1] - ry)
            > self.min_goal_distance
        ]

        self.publish_markers(candidates)

        if not candidates:
            if self.clusters:
                self.get_logger().info(
                    "%d frontiers left but all blacklisted or too close - "
                    "exploration complete." % len(self.clusters))
            else:
                self.get_logger().info("No frontiers left - map complete.")
            self.cmd_pub.publish(Twist())
            self.set_state(State.DONE, "no reachable frontiers")
            return

        if self.goal_pub.get_subscription_count() == 0:
            self.get_logger().warn(
                "No subscriber on the goal topic yet - waiting for "
                "a_star_planner before committing to a frontier.",
                throttle_duration_sec=5.0)
            return

        best = score_clusters(candidates, rx, ry, self.strategy,
                              self.size_weight)

        self.goal = best["world"]
        self.goal_start_time = now
        self.goal_start_pose = (rx, ry)

        self.publish_goal(rx, ry)
        self.last_republish = now

        self.get_logger().info(
            "Goal (%+.2f, %+.2f)  cluster=%d cells  clearance=%.2f m  "
            "dist=%.2f m  (%d candidates)"
            % (self.goal[0], self.goal[1], best["size"], best["clearance"],
               best["dist"], len(candidates)))

        self.set_state(State.NAVIGATING)

    # ------------------------------------------------------------------

    def publish_markers(self, candidates):
        arr = MarkerArray()

        clear = Marker()
        clear.header.frame_id = self.global_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        m = Marker()
        m.header.frame_id = self.global_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "frontiers"
        m.id = 0
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.pose.orientation.w = 1.0
        for f in candidates:
            pt = Point()
            pt.x, pt.y = float(f["world"][0]), float(f["world"][1])
            pt.z = 0.1
            m.points.append(pt)
            col = ColorRGBA()
            col.r, col.g, col.b, col.a = 0.1, 0.8, 0.2, 0.9
            m.colors.append(col)
        arr.markers.append(m)

        if self.blacklist:
            b = Marker()
            b.header.frame_id = self.global_frame
            b.header.stamp = m.header.stamp
            b.ns = "blacklist"
            b.id = 1
            b.type = Marker.SPHERE_LIST
            b.action = Marker.ADD
            b.scale.x = b.scale.y = b.scale.z = 0.25
            b.pose.orientation.w = 1.0
            b.color.r, b.color.g, b.color.b, b.color.a = 0.9, 0.1, 0.1, 0.7
            for bx, by in self.blacklist:
                pt = Point()
                pt.x, pt.y, pt.z = float(bx), float(by), 0.1
                b.points.append(pt)
            arr.markers.append(b)

        if self.goal is not None:
            t = Marker()
            t.header.frame_id = self.global_frame
            t.header.stamp = m.header.stamp
            t.ns = "target"
            t.id = 2
            t.type = Marker.SPHERE
            t.action = Marker.ADD
            t.scale.x = t.scale.y = t.scale.z = 0.35
            t.pose.position.x = float(self.goal[0])
            t.pose.position.y = float(self.goal[1])
            t.pose.position.z = 0.15
            t.pose.orientation.w = 1.0
            t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 0.6, 0.0, 1.0
            arr.markers.append(t)

        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
