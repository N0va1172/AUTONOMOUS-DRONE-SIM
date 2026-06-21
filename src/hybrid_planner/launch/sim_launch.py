import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('hybrid_planner')
    rviz_config = os.path.join(pkg_share, 'rviz', 'planner_view.rviz')

    start_x = LaunchConfiguration('start_x')
    start_y = LaunchConfiguration('start_y')
    goal_x = LaunchConfiguration('goal_x')
    goal_y = LaunchConfiguration('goal_y')
    flight_altitude = LaunchConfiguration('flight_altitude')

    return LaunchDescription([
        DeclareLaunchArgument('start_x', default_value='1.0'),
        DeclareLaunchArgument('start_y', default_value='1.0'),
        DeclareLaunchArgument('goal_x', default_value='9.0'),
        DeclareLaunchArgument('goal_y', default_value='9.0'),
        DeclareLaunchArgument('flight_altitude', default_value='3.0'),

        Node(
            package='hybrid_planner',
            executable='planner_node',
            name='hybrid_planner_node',
            output='screen',
            parameters=[{
                'start_x': start_x,
                'start_y': start_y,
                'goal_x': goal_x,
                'goal_y': goal_y,
                'flight_altitude': flight_altitude,
                'min_rand': 0.0,
                'max_rand': 10.0,
                'expand_dis': 0.5,
                'goal_sample_rate': 5,
                'max_iter': 3000,
                'k_neighbors': 10,
                'frame_id': 'map',
            }],
        ),

        Node(
            package='hybrid_planner',
            executable='drone_commander',
            name='drone_commander',
            output='screen',
            parameters=[{
                'flight_altitude': flight_altitude,
                'waypoint_tolerance': 0.4,
            }],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])
