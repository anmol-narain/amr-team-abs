import os
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # --- Package Name ---
    pkg_name = 'amr-team-abs'

    # --- RViz Configuration ---
    # Using the same config from Task 1, but you can save a specific task3.rviz later
    rviz_config_path = os.path.expanduser('~/.rviz2/task1.rviz')

    # --- Step 1: The Brain & Mapper (Our Unified Node) ---
    # This replaces slam_toolbox, the static map server, and the Task 2 Particle Filter
    exploration_mapper_node = Node(
        package=pkg_name,
        executable='exploration_mapper',
        name='exploration_mapper',
        output='screen'
    )

    # --- Step 2: RViz2 Visualization ---
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    # --- Step 3: Navigation Planners (With a 3-Second Delay) ---
    # We delay the planners slightly so the mapper has time to publish the first blank grid
    delayed_planners = TimerAction(
        period=16.0,
        actions=[
            Node(
                package=pkg_name,
                executable='a_star_planner',
                name='a_star_planner',
                output='screen'
            ),
            Node(
                package=pkg_name,
                executable='potential_field_planner',
                name='potential_field_planner',
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        exploration_mapper_node,
        rviz_node,
        delayed_planners
    ])
