"""Task 3: frontier exploration with SLAM Toolbox.

  slam_toolbox      -> /map  and  TF map->odom     (the SLAM component)
  frontier_explorer -> /goal_pose                  (our contribution)
  a_star_planner    -> /global_path
  potential_field   -> /cmd_vel

SLAM Toolbox runs with its OWN bundled default parameters
(mapper_params_online_async.yaml, inside the slam_toolbox package). We
include its launch file rather than starting the node directly, so
there is no parameter file of ours to install or keep in sync.

Startup order matters, so the nodes are staged:

  t=0   slam_toolbox starts mapping from the first scan
  t=0   frontier_explorer starts and spins in place for 8 s, so SLAM
        sees in every direction before we commit to a first frontier
  t=12  A* and the potential field planner start - after the spin, so
        only one node publishes /cmd_vel at a time

DO NOT also run particle_filter or map_publisher. slam_toolbox owns
/map and map->odom now; a second publisher on that TF edge makes the
tree flicker and both planners read garbage poses.

USAGE
  ros2 launch amr-team-abs task3_exploration.launch.py
  ros2 launch amr-team-abs task3_exploration.launch.py strategy:=nearest
  ros2 launch amr-team-abs task3_exploration.launch.py explore:=false
  ros2 launch amr-team-abs task3_exploration.launch.py mode:=sim

TUNING SLAM WITHOUT A PARAMS FILE
  The default minimum_travel_distance is 0.5 m, which is coarse for a
  small lab. Change it live, no rebuild:
      ros2 param set /slam_toolbox minimum_travel_distance 0.3

SAVE THE MAP when exploration finishes:
  ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
    "{name: {data: '/home/mik/ros2_ws/src/amr-team-abs/maps/explored_map'}}"

Then close the loop with Task 2 on the map you just built:
  ros2 launch amr-team-abs task1_navigation.launch.py localisation:=mcl \
    map:=$HOME/ros2_ws/src/amr-team-abs/maps/explored_map.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'amr-team-abs'


def resolve_rviz(share, override):
    for c in (override,
              os.path.join(share, "config", "task3.rviz"),
              os.path.join(share, "config", "task1.rviz"),
              os.path.expanduser("~/.rviz2/task1.rviz")):
        if c and os.path.exists(c):
            return c
    return None


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory(PKG)
    lc = lambda k: LaunchConfiguration(k).perform(context)

    mode = lc("mode")
    use_sim_time = (mode == "sim")
    common = {"use_sim_time": use_sim_time}

    want_explore = lc("explore").lower() == "true"
    want_planners = lc("planners").lower() == "true"
    want_rviz = lc("rviz").lower() == "true"
    strategy = lc("strategy")
    delay = float(lc("planner_delay"))
    spin = float(lc("spin_duration"))

    if strategy not in ("nearest", "largest", "cost_utility"):
        raise RuntimeError(
            "strategy must be nearest, largest or cost_utility - got '%s'"
            % strategy)

    try:
        slam_share = get_package_share_directory("slam_toolbox")
    except Exception:
        raise RuntimeError(
            "slam_toolbox is not installed:\n"
            "  sudo apt install ros-humble-slam-toolbox")

    slam_launch = os.path.join(slam_share, "launch", "online_async_launch.py")
    if not os.path.exists(slam_launch):
        raise RuntimeError("slam_toolbox launch file missing: %s" % slam_launch)

    rviz_cfg = resolve_rviz(share, lc("rviz_config"))

    print("[task3] mode        = %s" % mode)
    print("[task3] slam        = slam_toolbox defaults (%s)" % slam_launch)
    print("[task3] strategy    = %s" % strategy)
    print("[task3] rviz config = %s" % (rviz_cfg or "none (defaults)"))

    nodes = [
        # SLAM Toolbox with its own bundled parameters. Publishes /map
        # and the corrected map->odom transform.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch),
            launch_arguments={
                "use_sim_time": str(use_sim_time).lower(),
            }.items(),
        ),
    ]

    if want_rviz:
        nodes.append(Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=(["-d", rviz_cfg] if rviz_cfg else []),
            output="screen", parameters=[common]))

    # Starts immediately so its initial spin overlaps SLAM warming up.
    if want_explore:
        nodes.append(Node(
            package=PKG, executable="frontier_explorer",
            name="frontier_explorer", output="screen",
            parameters=[{
                "selection_strategy": strategy,
                "size_weight": float(lc("size_weight")),
                "min_cluster_cells": int(lc("min_cluster_cells")),
                "robot_radius": float(lc("robot_radius")),
                "initial_spin": lc("initial_spin").lower() == "true",
                "spin_duration": spin,
                "goal_timeout": float(lc("goal_timeout")),
                "use_sim_time": use_sim_time,
            }]))

    # Delayed until after the spin, so only one node publishes /cmd_vel
    # at a time, and so /map exists before A* asks TF for the robot pose.
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
                              description="sim or real; sets use_sim_time"),
        DeclareLaunchArgument("strategy", default_value="cost_utility",
                              description="nearest, largest or cost_utility"),
        DeclareLaunchArgument("size_weight", default_value="3.0"),
        DeclareLaunchArgument("min_cluster_cells", default_value="8"),
        DeclareLaunchArgument("robot_radius", default_value="0.25"),
        DeclareLaunchArgument("goal_timeout", default_value="25.0"),
        DeclareLaunchArgument("initial_spin", default_value="true"),
        DeclareLaunchArgument("spin_duration", default_value="8.0"),
        DeclareLaunchArgument("planner_delay", default_value="12.0"),
        DeclareLaunchArgument("explore", default_value="true",
                              description="false = SLAM only, drive by teleop"),
        DeclareLaunchArgument("planners", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=""),
        OpaqueFunction(function=launch_setup),
    ])