from setuptools import setup

setup(
    name='gym_nav2d',
    version='0.1',
    description='Navigation domain OpenAI Gym environment',
    author='Cheng Liu',
    packages=['gym_nav2d'],
    install_requires=['gym', 'numpy', 'pygame']
)