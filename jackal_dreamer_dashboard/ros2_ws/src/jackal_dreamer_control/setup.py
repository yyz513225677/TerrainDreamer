from setuptools import setup

package_name = 'jackal_dreamer_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leo',
    maintainer_email='yyz513225677@gmail.com',
    description='Mode FSM and manual control nodes for the Jackal-Dreamer dashboard.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mode_manager = jackal_dreamer_control.mode_manager:main',
            'manual_control = jackal_dreamer_control.manual_control:main',
        ],
    },
)
