# gym-nav2D-hybrid

A UAV 2D navigation environment with hybrid action space built on OpenAI gym.

## Dependencies

+ Python 3.5+
+ gym 0.10.5
+ pygame 1.9.4
+ numpy

## Installation

```shell
git clone https://github.com/chengliu-LR/gym-nav2D-hybrid.git
cd gym-nav2D-hybrid
python3 -m pip install -e '.[gym-nav2D-hybrid]'
```

## Demo

```python
import gym_nav2d
env = gym.make('nav2d-v0') # env id
```
