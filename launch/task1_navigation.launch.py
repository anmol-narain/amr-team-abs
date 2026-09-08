import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # --- File Paths ---
    map_path = '/home/anmol/ros2_amr/src/amr-team-abs/maps/my_lab_map5new.yaml'

    # Using os.path.expanduser to safely resolve the "~/.rviz2" hidden folder
    rviz_config_path = os.path.expanduser('~/.rviz2/task1.rviz')

    # --- Step 1: Map Server & AMCL (Localization) ---
    # This automatically handles the map_server and lifecycle nodes for you!
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'localization_launch.py')
        ),
        launch_arguments={'map': map_path}.items()
    )

    # --- Step 2: RViz2 ---
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    # --- Step 3: Custom Planners (With an 8-Second Delay) ---
    # We delay the brain and driver so the map and TF tree have time to initialize
    delayed_planners = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='amr-team-abs',
                executable='a_star_planner',
                name='a_star_planner',
                output='screen'
            ),
            Node(
                package='amr-team-abs',
                executable='potential_field_planner',
                name='potential_field_planner',
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        localization_launch,
        rviz_node,
        delayed_planners
    ])
