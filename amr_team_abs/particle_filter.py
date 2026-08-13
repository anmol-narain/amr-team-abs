#!/usr/bin/env python3
"""Monte Carlo Localisation (particle filter) for the Robile.

  in : /map (OccupancyGrid), /scan (LaserScan), TF odom->base_link,
       /initialpose (RViz "2D Pose Estimate")
  out: TF map->odom, /particle_cloud (PoseArray), /mcl_pose

  1. predict  - odometry motion model with noise
  2. correct  - likelihood field sensor model
  3. resample - low-variance systematic, only when Neff < half
"""

import math
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import (Pose, PoseArray, PoseWithCovarianceStamped,
                               TransformStamped)
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

import tf2_ros


def normalize_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


def quat_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_likelihood_field(occupied, resolution, max_dist):
    """Distance in metres from every cell to the nearest occupied cell."""
    from scipy.ndimage import distance_transform_edt
    if not occupied.any():
        return np.full(occupied.shape, max_dist, dtype=np.float32)
    dist_cells = distance_transform_edt(~occupied)
    dist_m = dist_cells.astype(np.float32) * float(resolution)
    return np.minimum(dist_m, float(max_dist))


def sample_motion_odometry(particles, d_rot1, d_trans, d_rot2, alphas, rng):
    """Textbook rot1/trans/rot2 model for a differential drive."""
    a1, a2, a3, a4 = alphas
    n = particles.shape[0]
    s_rot1 = math.sqrt(a1 * d_rot1 * d_rot1 + a2 * d_trans * d_trans)
    s_trans = math.sqrt(a3 * d_trans * d_trans
                        + a4 * (d_rot1 * d_rot1 + d_rot2 * d_rot2))
    s_rot2 = math.sqrt(a1 * d_rot2 * d_rot2 + a2 * d_trans * d_trans)
    rot1 = d_rot1 - rng.normal(0.0, s_rot1, n) if s_rot1 > 0 else np.full(n, d_rot1)
    trans = d_trans - rng.normal(0.0, s_trans, n) if s_trans > 0 else np.full(n, d_trans)
    rot2 = d_rot2 - rng.normal(0.0, s_rot2, n) if s_rot2 > 0 else np.full(n, d_rot2)
    th = particles[:, 2]
    out = np.empty_like(particles)
    out[:, 0] = particles[:, 0] + trans * np.cos(th + rot1)
    out[:, 1] = particles[:, 1] + trans * np.sin(th + rot1)
    out[:, 2] = normalize_angle(th + rot1 + rot2)
    return out


def sample_motion_omni(particles, dx_local, dy_local, dtheta, alphas, rng):
    """Motion model for an omnidirectional base - this is the Robile.

    rot1/trans/rot2 assumes the robot only drives along its own x axis.
    The Robile can strafe, so a sideways move would be modelled as a
    huge rotate-drive-rotate and inject nonsense noise.
    """
    a_tt, a_tr, a_rt, a_rr = alphas
    n = particles.shape[0]
    d = math.hypot(dx_local, dy_local)
    s_trans = a_tt * d + a_tr * abs(dtheta)
    s_rot = a_rr * abs(dtheta) + a_rt * d
    dx = dx_local + (rng.normal(0.0, s_trans, n) if s_trans > 0 else 0.0)
    dy = dy_local + (rng.normal(0.0, s_trans, n) if s_trans > 0 else 0.0)
    dth = dtheta + (rng.normal(0.0, s_rot, n) if s_rot > 0 else 0.0)
    th = particles[:, 2]
    cos_t, sin_t = np.cos(th), np.sin(th)
    out = np.empty_like(particles)
    out[:, 0] = particles[:, 0] + cos_t * dx - sin_t * dy
    out[:, 1] = particles[:, 1] + sin_t * dx + cos_t * dy
    out[:, 2] = normalize_angle(th + dth)
    return out


def likelihood_field_weights(particles, ranges, angles, laser_pose,
                             field, origin, resolution, max_dist,
                             z_hit, z_rand, sigma_hit, softening):
    """Project every beam into the map per particle, score by distance
    to the nearest wall. `softening` flattens the summed log-likelihood
    so one particle cannot take all the weight."""
    n = particles.shape[0]
    if ranges.size == 0:
        return np.full(n, 1.0 / n)

    lx, ly, lyaw = laser_pose
    px = particles[:, 0:1]
    py = particles[:, 1:2]
    pth = particles[:, 2:3]

    cos_p, sin_p = np.cos(pth), np.sin(pth)
    ox_l = px + cos_p * lx - sin_p * ly
    oy_l = py + sin_p * lx + cos_p * ly
    th_l = pth + lyaw

    a = th_l + angles[None, :]
    ex = ox_l + ranges[None, :] * np.cos(a)
    ey = oy_l + ranges[None, :] * np.sin(a)

    h, w = field.shape
    col = ((ex - origin[0]) / resolution).astype(np.int32)
    row = ((ey - origin[1]) / resolution).astype(np.int32)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    np.clip(col, 0, w - 1, out=col)
    np.clip(row, 0, h - 1, out=row)

    dist = field[row, col]
    dist = np.where(inside, dist, max_dist)

    p = (z_hit * np.exp(-(dist * dist) / (2.0 * sigma_hit * sigma_hit))
         + z_rand / max_dist)

    log_w = np.log(np.maximum(p, 1e-12)).sum(axis=1) * softening
    log_w -= log_w.max()
    w_out = np.exp(log_w)
    total = w_out.sum()
    return w_out / total if total > 0 else np.full(n, 1.0 / n)


def effective_sample_size(weights):
    return 1.0 / np.sum(weights * weights)


def systematic_resample(weights, n_out, rng):
    positions = (np.arange(n_out) + rng.random()) / n_out
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions).clip(0, weights.size - 1)


def kld_sample_size(bins, epsilon, z_delta):
    k = len(bins)
    if k <= 1:
        return 0
    a = 2.0 / (9.0 * (k - 1))
    return int(math.ceil((k - 1) / (2.0 * epsilon)
                         * (1.0 - a + math.sqrt(a) * z_delta) ** 3))


def kld_resample(particles, weights, rng, min_particles, max_particles,
                 bin_xy, bin_theta, epsilon, z_delta):
    """Adaptive MCL: keep drawing until the KLD bound is met."""
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    out = []
    bins = set()
    needed = max_particles
    while len(out) < max_particles:
        i = int(np.searchsorted(cumsum, rng.random()))
        i = min(i, particles.shape[0] - 1)
        p = particles[i]
        out.append(p)
        key = (int(math.floor(p[0] / bin_xy)),
               int(math.floor(p[1] / bin_xy)),
               int(math.floor(p[2] / bin_theta)))
        if key not in bins:
            bins.add(key)
            needed = kld_sample_size(bins, epsilon, z_delta)
        if len(out) >= max(needed, min_particles):
            break
    return np.array(out, dtype=np.float64)


def weighted_mean_pose(particles, weights):
    x = float(np.sum(weights * particles[:, 0]))
    y = float(np.sum(weights * particles[:, 1]))
    c = float(np.sum(weights * np.cos(particles[:, 2])))
    s = float(np.sum(weights * np.sin(particles[:, 2])))
    return x, y, math.atan2(s, c)


def pose_covariance(particles, weights, mean):
    dx = particles[:, 0] - mean[0]
    dy = particles[:, 1] - mean[1]
    dt = normalize_angle(particles[:, 2] - mean[2])
    cov = np.zeros((3, 3))
    cov[0, 0] = np.sum(weights * dx * dx)
    cov[1, 1] = np.sum(weights * dy * dy)
    cov[2, 2] = np.sum(weights * dt * dt)
    cov[0, 1] = cov[1, 0] = np.sum(weights * dx * dy)
    return cov


def compose_map_to_odom(map_base, odom_base):
    """map->odom = (map->base) * (odom->base)^-1. The whole point."""
    mx, my, mth = map_base
    ox, oy, oth = odom_base
    th = normalize_angle(mth - oth)
    x = mx - (math.cos(th) * ox - math.sin(th) * oy)
    y = my - (math.sin(th) * ox + math.cos(th) * oy)
    return x, y, th


def scatter_uniform(free_cells, origin, resolution, n, rng):
    idx = rng.integers(0, free_cells.shape[0], n)
    rows = free_cells[idx, 0] + rng.random(n)
    cols = free_cells[idx, 1] + rng.random(n)
    p = np.empty((n, 3))
    p[:, 0] = origin[0] + cols * resolution
    p[:, 1] = origin[1] + rows * resolution
    p[:, 2] = rng.uniform(-np.pi, np.pi, n)
    return p


def scatter_gaussian(x, y, yaw, sx, sy, syaw, n, rng):
    p = np.empty((n, 3))
    p[:, 0] = rng.normal(x, sx, n)
    p[:, 1] = rng.normal(y, sy, n)
    p[:, 2] = normalize_angle(rng.normal(yaw, syaw, n))
    return p


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")
        p = self.declare_parameter

        p("global_frame", "map")
        p("odom_frame", "odom")
        p("base_frame", "base_link")
        p("map_topic", "/map")
        p("scan_topic", "/scan")

        p("num_particles", 500)
        p("use_kld_sampling", False)
        p("min_particles", 200)
        p("max_particles", 2000)
        p("kld_epsilon", 0.05)
        p("kld_z", 2.33)
        p("kld_bin_xy", 0.5)
        p("kld_bin_theta", 0.26)

        p("motion_model", "omni")
        p("alpha1", 0.2)
        p("alpha2", 0.2)
        p("alpha3", 0.2)
        p("alpha4", 0.2)
        p("omni_alpha_trans_trans", 0.15)
        p("omni_alpha_trans_rot", 0.05)
        p("omni_alpha_rot_trans", 0.05)
        p("omni_alpha_rot_rot", 0.15)

        p("max_beams", 60)
        p("z_hit", 0.85)
        p("z_rand", 0.15)
        p("sigma_hit", 0.20)
        p("likelihood_max_dist", 2.0)
        p("weight_softening", 0.3)
        p("laser_max_range", 8.0)
        p("laser_min_range", 0.10)

        p("update_min_d", 0.20)
        p("update_min_a", 0.20)
        p("resample_neff_ratio", 0.5)

        p("global_localisation", False)
        p("initial_pose_x", 0.0)
        p("initial_pose_y", 0.0)
        p("initial_pose_yaw", 0.0)
        p("initial_cov_xy", 0.5)
        p("initial_cov_yaw", 0.3)

        p("tf_publish_rate", 20.0)
        p("transform_tolerance", 0.2)
        p("map_yaml", "")

        g = lambda k: self.get_parameter(k).value
        self.global_frame = g("global_frame")
        self.odom_frame = g("odom_frame")
        self.base_frame = g("base_frame")
        self.num_particles = int(g("num_particles"))
        self.use_kld = bool(g("use_kld_sampling"))
        self.min_particles = int(g("min_particles"))
        self.max_particles = int(g("max_particles"))
        self.kld_epsilon = float(g("kld_epsilon"))
        self.kld_z = float(g("kld_z"))
        self.kld_bin_xy = float(g("kld_bin_xy"))
        self.kld_bin_theta = float(g("kld_bin_theta"))
        self.motion_model = g("motion_model")
        self.diff_alphas = (g("alpha1"), g("alpha2"), g("alpha3"), g("alpha4"))
        self.omni_alphas = (g("omni_alpha_trans_trans"), g("omni_alpha_trans_rot"),
                            g("omni_alpha_rot_trans"), g("omni_alpha_rot_rot"))
        self.max_beams = int(g("max_beams"))
        self.z_hit = float(g("z_hit"))
        self.z_rand = float(g("z_rand"))
        self.sigma_hit = float(g("sigma_hit"))
        self.max_dist = float(g("likelihood_max_dist"))
        self.softening = float(g("weight_softening"))
        self.laser_max_range = float(g("laser_max_range"))
        self.laser_min_range = float(g("laser_min_range"))
        self.update_min_d = float(g("update_min_d"))
        self.update_min_a = float(g("update_min_a"))
        self.neff_ratio = float(g("resample_neff_ratio"))
        self.tf_tolerance = float(g("transform_tolerance"))

        self.rng = np.random.default_rng()
        self.particles = None
        self.weights = None
        self.map_info = None
        self.field = None
        self.free_cells = None
        self.laser_pose = None
        self.last_odom = None
        self.map_to_odom = (0.0, 0.0, 0.0)
        self.updates = 0

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        sensor_qos = QoSProfile(depth=5,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(OccupancyGrid, g("map_topic"),
                                 self.map_callback, latched)
        self.create_subscription(LaserScan, g("scan_topic"),
                                 self.scan_callback, sensor_qos)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                 self.initialpose_callback, 10)

        self.cloud_pub = self.create_publisher(PoseArray, "/particle_cloud", 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                              "/mcl_pose", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_service(Empty, "global_localization",
                            self.global_localisation_srv)

        rate = float(g("tf_publish_rate"))
        self.create_timer(1.0 / max(rate, 1.0), self.publish_transform)

        yaml_path = g("map_yaml")
        if yaml_path:
            self.load_map_from_file(yaml_path)

        self.get_logger().info(
            "MCL up. model=%s particles=%s. Waiting for map + scan."
            % (self.motion_model,
               ("KLD %d-%d" % (self.min_particles, self.max_particles))
               if self.use_kld else self.num_particles))

    def map_callback(self, msg):
        if self.field is not None:
            return
        info = {
            "resolution": msg.info.resolution,
            "origin": (msg.info.origin.position.x, msg.info.origin.position.y),
            "width": msg.info.width,
            "height": msg.info.height,
        }
        # OccupancyGrid.data is ALREADY row 0 = bottom. Do NOT flipud.
        grid = np.array(msg.data, dtype=np.int8).reshape(
            (info["height"], info["width"]))
        self.set_map(grid, info)

    def load_map_from_file(self, yaml_path):
        import yaml
        from PIL import Image
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img_path = meta["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)),
                                    img_path)
        # .pgm row 0 is the TOP; the map frame origin is BOTTOM-left.
        pixels = np.flipud(np.array(Image.open(img_path), dtype=np.float64))
        p = pixels / 255.0 if int(meta.get("negate", 0)) else (255.0 - pixels) / 255.0
        grid = np.full(p.shape, -1, dtype=np.int8)
        grid[p > float(meta.get("occupied_thresh", 0.65))] = 100
        grid[p < float(meta.get("free_thresh", 0.25))] = 0
        info = {
            "resolution": float(meta["resolution"]),
            "origin": (float(meta["origin"][0]), float(meta["origin"][1])),
            "width": grid.shape[1],
            "height": grid.shape[0],
        }
        self.set_map(grid, info)

    def set_map(self, grid, info):
        self.map_info = info
        occupied = (grid == 100)
        self.field = build_likelihood_field(occupied, info["resolution"],
                                            self.max_dist)
        self.free_cells = np.argwhere(grid == 0)
        w, h, r = info["width"], info["height"], info["resolution"]
        ox, oy = info["origin"]
        self.get_logger().info(
            "Map: %dx%d cells @ %s m -> %.2f x %.2f m, x %.2f..%.2f, "
            "y %.2f..%.2f, free=%d occ=%d"
            % (w, h, r, w * r, h * r, ox, ox + w * r, oy, oy + h * r,
               len(self.free_cells), int(occupied.sum())))
        if self.particles is None:
            self.initialise_particles()

    def initialise_particles(self):
        n = self.max_particles if self.use_kld else self.num_particles
        if self.get_parameter("global_localisation").value:
            if self.free_cells is None or len(self.free_cells) == 0:
                return
            self.particles = scatter_uniform(
                self.free_cells, self.map_info["origin"],
                self.map_info["resolution"], n, self.rng)
            self.get_logger().info(
                "Global localisation: %d particles scattered." % n)
        else:
            x = float(self.get_parameter("initial_pose_x").value)
            y = float(self.get_parameter("initial_pose_y").value)
            yaw = float(self.get_parameter("initial_pose_yaw").value)
            sxy = float(self.get_parameter("initial_cov_xy").value)
            syaw = float(self.get_parameter("initial_cov_yaw").value)
            self.particles = scatter_gaussian(x, y, yaw, sxy, sxy, syaw, n,
                                              self.rng)
            self.get_logger().info(
                "Initial pose (%.2f, %.2f, %.0f deg), %d particles."
                % (x, y, math.degrees(yaw), n))
        self.weights = np.full(self.particles.shape[0],
                               1.0 / self.particles.shape[0])
        self.last_odom = None

    def initialpose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        sx = math.sqrt(max(msg.pose.covariance[0], 0.01))
        sy = math.sqrt(max(msg.pose.covariance[7], 0.01))
        syaw = math.sqrt(max(msg.pose.covariance[35], 0.01))
        n = self.max_particles if self.use_kld else self.num_particles
        self.particles = scatter_gaussian(x, y, yaw, sx, sy, syaw, n, self.rng)
        self.weights = np.full(n, 1.0 / n)
        self.last_odom = None
        self.get_logger().info("Pose reset to (%.2f, %.2f, %.0f deg)"
                               % (x, y, math.degrees(yaw)))

    def global_localisation_srv(self, request, response):
        if self.free_cells is not None and len(self.free_cells):
            n = self.max_particles if self.use_kld else self.num_particles
            self.particles = scatter_uniform(
                self.free_cells, self.map_info["origin"],
                self.map_info["resolution"], n, self.rng)
            self.weights = np.full(n, 1.0 / n)
            self.last_odom = None
            self.get_logger().info("Kidnapped: particles scattered globally.")
        return response

    def lookup_odom(self, stamp):
        try:
            t = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn("odom->base TF unavailable: %s" % e,
                                   throttle_duration_sec=5.0)
            return None
        q = t.transform.rotation
        return (t.transform.translation.x, t.transform.translation.y,
                quat_to_yaw(q.x, q.y, q.z, q.w))

    def lookup_laser_pose(self, frame_id):
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn("base->%s TF unavailable: %s" % (frame_id, e),
                                   throttle_duration_sec=5.0)
            return None
        q = t.transform.rotation
        pose = (t.transform.translation.x, t.transform.translation.y,
                quat_to_yaw(q.x, q.y, q.z, q.w))
        self.get_logger().info(
            "Laser '%s' at (%.3f, %.3f, %.1f deg) in %s"
            % (frame_id, pose[0], pose[1], math.degrees(pose[2]),
               self.base_frame))
        return pose

    def scan_callback(self, scan):
        if self.field is None or self.particles is None:
            return

        if self.laser_pose is None:
            self.laser_pose = self.lookup_laser_pose(scan.header.frame_id)
            if self.laser_pose is None:
                return

        odom = self.lookup_odom(scan.header.stamp)
        if odom is None:
            return

        if self.last_odom is None:
            self.last_odom = odom
            self.publish_cloud(scan.header.stamp)
            return

        dx = odom[0] - self.last_odom[0]
        dy = odom[1] - self.last_odom[1]
        dth = normalize_angle(odom[2] - self.last_odom[2])
        dist = math.hypot(dx, dy)

        if dist < self.update_min_d and abs(dth) < self.update_min_a:
            return

        if self.motion_model == "diff":
            d_rot1 = normalize_angle(math.atan2(dy, dx) - self.last_odom[2]) \
                if dist > 0.01 else 0.0
            d_rot2 = normalize_angle(dth - d_rot1)
            self.particles = sample_motion_odometry(
                self.particles, d_rot1, dist, d_rot2, self.diff_alphas, self.rng)
        else:
            c, s = math.cos(self.last_odom[2]), math.sin(self.last_odom[2])
            dx_l = c * dx + s * dy
            dy_l = -s * dx + c * dy
            self.particles = sample_motion_omni(
                self.particles, dx_l, dy_l, dth, self.omni_alphas, self.rng)

        self.last_odom = odom

        ranges, angles = self.prepare_scan(scan)
        self.weights = likelihood_field_weights(
            self.particles, ranges, angles, self.laser_pose,
            self.field, self.map_info["origin"], self.map_info["resolution"],
            self.max_dist, self.z_hit, self.z_rand, self.sigma_hit,
            self.softening)

        n = self.particles.shape[0]
        neff = effective_sample_size(self.weights)
        if neff < self.neff_ratio * n:
            if self.use_kld:
                self.particles = kld_resample(
                    self.particles, self.weights, self.rng,
                    self.min_particles, self.max_particles,
                    self.kld_bin_xy, self.kld_bin_theta,
                    self.kld_epsilon, self.kld_z)
            else:
                idx = systematic_resample(self.weights, n, self.rng)
                self.particles = self.particles[idx]
            m = self.particles.shape[0]
            self.weights = np.full(m, 1.0 / m)

        mean = weighted_mean_pose(self.particles, self.weights)
        odom_now = self.lookup_odom(scan.header.stamp) or odom
        self.map_to_odom = compose_map_to_odom(mean, odom_now)

        self.publish_pose(scan.header.stamp, mean)
        self.publish_cloud(scan.header.stamp)

        self.updates += 1
        if self.updates % 10 == 1:
            self.get_logger().info(
                "pose (%+.2f, %+.2f, %+.0f deg)  N=%d  Neff=%.0f  beams=%d"
                % (mean[0], mean[1], math.degrees(mean[2]),
                   self.particles.shape[0], neff, ranges.size))

    def prepare_scan(self, scan):
        raw = np.asarray(scan.ranges, dtype=np.float64)
        step = max(1, int(math.ceil(raw.size / max(self.max_beams, 1))))
        idx = np.arange(0, raw.size, step)
        r = raw[idx]
        a = scan.angle_min + idx * scan.angle_increment
        rmax = min(self.laser_max_range, scan.range_max)
        rmin = max(self.laser_min_range, scan.range_min)
        ok = np.isfinite(r) & (r > rmin) & (r < rmax)
        return r[ok], a[ok]

    def publish_transform(self):
        if self.map_info is None:
            return
        x, y, th = self.map_to_odom
        qx, qy, qz, qw = yaw_to_quat(th)
        t = TransformStamped()
        now = self.get_clock().now() + Duration(seconds=self.tf_tolerance)
        t.header.stamp = now.to_msg()
        t.header.frame_id = self.global_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def publish_pose(self, stamp, mean):
        cov = pose_covariance(self.particles, self.weights, mean)
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.global_frame
        msg.pose.pose.position.x = mean[0]
        msg.pose.pose.position.y = mean[1]
        q = yaw_to_quat(mean[2])
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        c = [0.0] * 36
        c[0] = cov[0, 0]
        c[1] = cov[0, 1]
        c[6] = cov[1, 0]
        c[7] = cov[1, 1]
        c[35] = cov[2, 2]
        msg.pose.covariance = c
        self.pose_pub.publish(msg)

    def publish_cloud(self, stamp):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.global_frame
        for px, py, pth in self.particles:
            pose = Pose()
            pose.position.x = float(px)
            pose.position.y = float(py)
            q = yaw_to_quat(float(pth))
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            msg.poses.append(pose)
        self.cloud_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
