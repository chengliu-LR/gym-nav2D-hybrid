import gym
import gym_nav2d

env = gym.make('nav2d-v0')
print('action space:\n', env.action_space)
print('observation space:\n', env.observation_space)
env.reset()

for i in range(1000):
    env.render()