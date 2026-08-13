import os
from glob import glob
from setuptools import setup

package_name = 'amr-team-abs'
module_name = 'amr_team_abs'

setup(
    name=package_name,
    version='0.0.0',
    packages=[module_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Tell colcon to install your launch files!
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Tell colcon to install your map files!
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@todo.todo',
    description='AMR Final Project Package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'potential_field_planner = amr_team_abs.potential_field_planner:main',
            'a_star_planner = amr_team_abs.a_star_planner:main',
            'particle_filter = amr_team_abs.particle_filter:main',
            'map_publisher = amr_team_abs.map_publisher:main',
        ],
    },
)
