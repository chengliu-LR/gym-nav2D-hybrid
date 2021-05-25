import gym
import math
import torch
import random
import gym_nav2d
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import namedtuple
#from torchviz import make_dot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.999
EPS_END = 0.001
EPS_DECAY = 500
TARGET_UPDATE = 50

# get the number of states from gym observation space
state_dim = 5
#get the number of actions from gym action space
n_actions = 3
#steps counter for epsilon annealing
steps_done = 0
num_episodes = 10
episode_rewards = np.zeros(num_episodes)

class FcDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FcDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, observation):
        qvals = self.fc_net(observation.view(-1, self.input_dim))
        return qvals

def select_action(state):
    global steps_done
    with torch.no_grad():
        #pick action with the largest expected reward
        return policy_net(state).argmax(dim=1, keepdim=True)

#initialize the networks, optimizers and the memory
policy_net = FcDQN(state_dim, n_actions).to(device)
policy_net.load_state_dict(torch.load('./trained_model/nav2d-trained-model.pt'))


# eval process
def evaluation(random_enabled):
    for epoch in range(num_episodes):
        env = gym.make('nav2d-v0')
        # env initial states
        initial_position = np.array([env.FIELD_HEIGHT / 2, env.FIELD_WIDTH / 2])
        state = torch.from_numpy(env.reset(initial_position, np.random.normal(loc=0, scale=2*np.pi))).unsqueeze(0).float().to(device)
        done = False
        #until the game is end
        while not done:
            #select and perform the action
            if not random_enabled:
                action = select_action(state)
            else:
                action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
            
            obs, reward, done, _ = env.step_discrete(action.item())
            if done:
                next_state = None
            else:
                next_state = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            episode_rewards[epoch] += reward
            reward = torch.tensor([[reward]], device=device)
            state = next_state
            env.render()

        #if epoch % 1 == 0:
        print('epoch:', epoch, 'accumulated reward:', episode_rewards[epoch])

# set random_enabled to True to see how random policy works
evaluation(random_enabled=True)
averaged_rewards = np.zeros(num_episodes)
averaged_rewards[:] = episode_rewards[:].sum() / num_episodes
print('Averaged reward among {} tests:'.format(num_episodes), episode_rewards[:].sum() / num_episodes)

#plot
plt.figure(dpi = 150) #set the resolution
plt.plot(episode_rewards, label='accumulated rewards')
plt.plot(averaged_rewards, label='averaged rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()
plt.title('DQN network test navigation')
plt.savefig("./figures/test_nav2d.png")
plt.close()

