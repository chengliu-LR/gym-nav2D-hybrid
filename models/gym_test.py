import gym
import gym_nav2d
import numpy as np
import pygame
#TODO: obstacle avoidance
env = gym.make('nav2d-v0')
#print('action space:\n', env.action_space)
#print('observation space:\n', env.observation_space)
num_epochs = 1
initial_position = np.array([env.FIELD_LENGTH / 2, env.FIELD_WIDTH / 2])
#initial_position = np.array([15., 15.])
initial_orientation = 0

for epochs in range(num_epochs):
    end_episode = False
    state, reward = env.reset(initial_position, initial_orientation)
    i = 0
    while not end_episode:
        i += 1
        if i % 2 == 0:
            action = (0, (np.random.randint(0,1), np.random.normal(loc=0, scale=2*np.pi)))
        else:
            action = (1, (np.random.randint(0,1), np.pi))
        #action = (0, (0.,0.))
        state, reward, end_episode, _ = env.step(action)
        env.render()

    print("end episode{}".format(epochs))

# env.reset(initial_position, initial_orientation)
# print(env.agent.position)
# action = (np.random.randint(0,2), (np.random.randint(0,1), np.random.normal()))
# state, reward, end_episode, _ = env.step(action)
# env.render()