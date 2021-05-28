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
EPS_DECAY = 200
TARGET_UPDATE = 50

state_dim = 5
n_actions = 2
# steps counter for epsilon annealing
steps_done = 0
num_episodes = 10
episode_rewards = np.zeros(num_episodes)

class PDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim))

    def forward(self, obs): # observation is cat([states, action_parameters])
        qvals = self.critic(obs.view(-1, self.input_dim))
        return qvals

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim),
            nn.Tanh()) # note that turning angle is set in [-pi, pi]
    
    def forward(self, states):
        continuous_act = self.action_bounds * self.actor(states.view(-1, self.state_dim))
        return continuous_act

def select_discrete_action(obs):
    return pdqn(obs).argmax(dim=1, keepdim=True)

# initialize the networks
env = gym.make('nav2d-v0')
action_bounds = torch.tensor(env.PARAMETERS_MAX, dtype=torch.float32).view(1,-1)
pdqn = PDQN(state_dim + n_actions, n_actions).to(device)
pdqn.load_state_dict(torch.load('./trained_model/trained_pdqn.pt'))
actor = Actor(state_dim, n_actions, action_bounds)
actor.load_state_dict(torch.load('./trained_model/trained_actor.pt'))

# eval process
def evaluation(random_enabled):
    for epoch in range(num_episodes):
        env = gym.make('nav2d-v0')
        # env initial states
        initial_position = np.array([env.FIELD_HEIGHT / 2, env.FIELD_WIDTH / 2])
        initial_orientation = -3 * np.pi / 4
        #initial_orientation = np.random.normal(loc=0, scale=2*np.pi)
        state = torch.from_numpy(env.reset(initial_position, initial_orientation)).unsqueeze(0).float().to(device)
        #state = torch.from_numpy(env.reset(initial_position, np.random.normal(loc=0, scale=2*np.pi))).unsqueeze(0).float().to(device)
        done = False
        # until the navigation is end
        while not done:
            # select and perform the action
            if not random_enabled:
                discrete_action = select_discrete_action(torch.cat([state, actor(state)], dim=1))
            else:
                discrete_action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
            
            obs, reward, done, _ = env.step((discrete_action.detach().item(), actor(state).detach().squeeze(0).numpy()))

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