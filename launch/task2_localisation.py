"""Full navigation stack for amr-team-abs.

  map_publisher     -> /map                (replaces nav2 map_server)
  particle_filter   -> TF map->odom        (replaces nav2 amcl)  [Task 2]
  a_star_planner    -> /global_path        [Task 1]
  potential_field_planner -> /cmd_vel      [Task 1]

WHAT CHANGED FROM THE PREVIOUS VERSION
--------------------------------------
1. nav2 removed. `get_package_share_directory('nav2_bringup')` raised on
   this machine - that is the `No module named nav2_common` error. Our
   own map_publisher and particle_filter do the same jobs and need
   nothing installed.
2. AMCL removed. The coursework asks you to "implement your very own
   particle filter", so AMCL cannot be the Task 2 deliverable anyway.
3. No absolute paths. The old file hardcoded a path under one person's
   home directory, so it only ran on one laptop. Everything now resolves
   from the installed package share, with launch-arg overrides.
4. MCL parameters are inline below, so this file works even if
   config/*.yaml has not been copied in yet. Pass params_file:=... to
   override with a YAML file once you have one.

USAGE
-----
  # your lab map, real robot or fake_robot_sim
  ros2 launch amr-team-abs task1_navigation.launch.py mode:=real

  # Gazebo with closed_walls
  ros2 launch amr-team-abs task1_navigation.launch.py mode:=sim

  # localisation only - always check this works before adding planners
  ros2 launch amr-team-abs task1_navigation.launch.py planners:=false

  # tell it where the robot is standing
  ros2 launch amr-team-abs task1_navigation.launch.py \
      mode:=real init_x:=-1.52 init_y:=-7.15 init_yaw:=0.0

Send a goal with RViz's 2D Goal Pose, or:
  ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
    "{header: {frame_id: map}, pose: {position: {x: 3.0, y: -6.0}, \
      orientation: {w: 1.0}}}"

NOTE: on my_lab_map5new every y coordinate is negative (-9.78 .. -2.98).
A goal with y = 0 is off the map and A* will refuse it.

DO NOT also run `static_transform_publisher 0 0 0 0 0 0 map odom` or
amcl. The particle filter owns the map->odom edge now; two publishers
make the TF tree flicker and both planners read garbage poses.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Package is registered with hyphens. That is illegal under REP-144;
# when you rename it to amr_team_abs, change this ONE line.
PKG = 'amr-team-abs'

# Default starting pose per map. my_lab_map5new: (-1.52, -7.15) is the
# most open point, 1.75 m from any wall. closed_walls: (3.0, -1.5) is
# the Gazebo spawn point.
DEFAULT_POSE = {
    "real": (-1.52, -7.15, 0.0),
    "sim": (3.0, -1.5, 0.0),
}


def resolve_map(mode, share, override):
    if override:
        return override
    if mode == "sim":
        try:
            nav = get_package_share_directory("robile_navigation")
            candidate = os.path.join(nav, "maps", "closed_walls.yaml")
            if os.path.exists(candidate):
                return candidate
        except Exception:
            pass
    return os.path.join(share, "maps", "my_lab_map5new.yaml")


def resolve_rviz(share, override):
    """~/.rviz2/task1.rviz is not in git, so a teammate who clones the
    repo will not have it. Commit the config to the package instead."""
    for c in (override,
              os.path.join(share, "config", "task1.rviz"),
              os.path.join(share, "rviz", "task1.rviz"),
              os.path.expanduser("~/.rviz2/task1.rviz")):
        if c and os.path.exists(c):
            return c
    return None


def mcl_parameters(mode, init_x, init_y, init_yaw, use_sim_time):
    """Inline MCL params. Keeps this launch file self-contained."""
    return {
        "use_sim_time": use_sim_time,

        "global_frame": "map",
        "odom_frame": "odom",
        "base_frame": "base_link",
        "map_topic": "/map",
        "scan_topic": "/scan",

        "num_particles": 500,
        "use_kld_sampling": False,
        "min_particles": 200,
        "max_particles": 2000,

        # Robile is omnidirectional, so "omni". "diff" gives the
        # textbook rot1/trans/rot2 model if you want to compare.
        "motion_model": "omni",
        # Real wheels slip more than simulated ones.
        "omni_alpha_trans_trans": 0.25 if mode == "real" else 0.15,
        "omni_alpha_trans_rot": 0.10 if mode == "real" else 0.05,
        "omni_alpha_rot_trans": 0.10 if mode == "real" else 0.05,
        "omni_alpha_rot_rot": 0.25 if mode == "real" else 0.15,

        "max_beams": 60,
        "z_hit": 0.75 if mode == "real" else 0.85,
        "z_rand": 0.25 if mode == "real" else 0.15,
        # Wider on hardware: tolerates map error and people walking past.
        "sigma_hit": 0.30 if mode == "real" else 0.20,
        "likelihood_max_dist": 2.0,
        "weight_softening": 0.3,
        "laser_max_range": 8.0,
        "laser_min_range": 0.10,

        "update_min_d": 0.20,
        "update_min_a": 0.20,
        "resample_neff_ratio": 0.5,

        "global_localisation": False,
        "initial_pose_x": init_x,
        "initial_pose_y": init_y,
        "initial_pose_yaw": init_yaw,
        "initial_cov_xy": 0.5,
        "initial_cov_yaw": 0.3,

        "tf_publish_rate": 20.0,
        "transform_tolerance": 0.3 if mode == "real" else 0.2,
    }


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory(PKG)
    lc = lambda k: LaunchConfiguration(k).perform(context)

    mode = lc("mode")
    map_yaml = resolve_map(mode, share, lc("map"))
    rviz_cfg = resolve_rviz(share, lc("rviz_config"))
    want_rviz = lc("rviz").lower() == "true"
    want_planners = lc("planners").lower() == "true"
    delay = float(lc("planner_delay"))
    params_file = lc("params_file")

    use_sim_time = (mode == "sim")

    dx, dy, dyaw = DEFAULT_POSE.get(mode, DEFAULT_POSE["real"])
    init_x = float(lc("init_x")) if lc("init_x") else dx
    init_y = float(lc("init_y")) if lc("init_y") else dy
    init_yaw = float(lc("init_yaw")) if lc("init_yaw") else dyaw

    if not os.path.exists(map_yaml):
        raise RuntimeError(
            "Map not found: %s\n"
            "Put the .pgm and .yaml in <repo>/maps/, make sure setup.py's "
            "data_files installs maps/*, rebuild, or pass "
            "map:=/full/path/to/map.yaml" % map_yaml)

    print("[task1_navigation] mode       = %s" % mode)
    print("[task1_navigation] map        = %s" % map_yaml)
    print("[task1_navigation] init pose  = (%.2f, %.2f, %.2f rad)"
          % (init_x, init_y, init_yaw))
    print("[task1_navigation] rviz cfg   = %s" % (rviz_cfg or "none (defaults)"))

    mcl_params = mcl_parameters(mode, init_x, init_y, init_yaw, use_sim_time)
    if params_file and os.path.exists(params_file):
        print("[task1_navigation] params     = %s (overrides inline)" % params_file)
        mcl_args = [params_file, {"use_sim_time": use_sim_time}]
    else:
        mcl_args = [mcl_params]

    common = {"use_sim_time": use_sim_time}

    nodes = [
        Node(package=PKG, executable="map_publisher", name="map_publisher",
             output="screen",
             parameters=[{"map_yaml": map_yaml, "use_sim_time": use_sim_time}]),

        Node(package=PKG, executable="particle_filter", name="particle_filter",
             output="screen", parameters=mcl_args),
    ]

    if want_rviz:
        nodes.append(Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=(["-d", rviz_cfg] if rviz_cfg else []),
            output="screen", parameters=[common]))

    # Delayed so /map is latched and the first map->odom exists before
    # A* asks TF where the robot is.
    if want_planners:
        nodes.append(TimerAction(period=delay, actions=[
            Node(package=PKG, executable="a_star_planner",
                 name="a_star_planner", output="screen", parameters=[common]),
            Node(package=PKG, executable="potential_field_planner",
                 name="potential_field_planner", output="screen",
                 parameters=[common]),
        ]))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="real",
                              description="sim (Gazebo/closed_walls) or real (lab map)"),
        DeclareLaunchArgument("map", default_value="",
                              description="full path to a map .yaml; blank = auto"),
        DeclareLaunchArgument("params_file", default_value="",
                              description="optional MCL params .yaml; blank = use inline defaults"),
        DeclareLaunchArgument("init_x", default_value=""),
        DeclareLaunchArgument("init_y", default_value=""),
        DeclareLaunchArgument("init_yaw", default_value="",
                              description="radians"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=""),
        DeclareLaunchArgument("planners", default_value="true",
                              description="false = map + MCL only"),
        DeclareLaunchArgument("planner_delay", default_value="6.0"),
        OpaqueFunction(function=launch_setup),
    ])
