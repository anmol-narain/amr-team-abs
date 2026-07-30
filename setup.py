from setuptools import setup

package_name = 'amr-team-abs'
module_name = 'amr_team_abs' # The inner folder with underscores

setup(
    name=package_name,
    version='0.0.0',
    packages=[module_name], # Point to the inner folder
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            # format: 'executable_name = module_folder.script_name:main'
            'potential_field_planner = amr_team_abs.potential_field_planner:main',
        ],
    },
)
