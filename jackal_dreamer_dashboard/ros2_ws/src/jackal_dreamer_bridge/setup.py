from setuptools import setup

package_name = 'jackal_dreamer_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'websockets'],
    zip_safe=True,
    maintainer='leo',
    maintainer_email='yyz513225677@gmail.com',
    description='Websocket multiplexer between the dashboard frontend and ROS 2.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dashboard_bridge = jackal_dreamer_bridge.dashboard_bridge:main',
        ],
    },
)
