from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Step 4: The Global Planner (A*)
        Node(
            package='amr-team-abs',
            executable='a_star_planner',
            name='a_star_planner',
            output='screen'
        ),
        # Step 5: The Local Planner (Potential Fields)
        Node(
            package='amr-team-abs',
            executable='potential_field_planner',
            name='potential_field_planner',
            output='screen'
        )
    ])
