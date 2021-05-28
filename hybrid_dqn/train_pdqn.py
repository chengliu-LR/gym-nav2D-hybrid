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

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.999
EPS_END = 1e-3
EPS_DECAY = 5e3
#TARGET_UPDATE = 50

# target network update rate
TAU = 5e-2

# get dim of states and actions from env space
env = gym.make('nav2d-v0')
state_dim = len(env.observation_space)
n_actions = len(env.action_space)
#steps counter for epsilon annealing
steps_done = 0
num_episodes = 20000

action_bounds = torch.tensor(env.PARAMETERS_MAX, dtype=torch.float32).view(1,-1)
gaussian_exploration_noise = 0.2

episode_rewards = np.zeros(num_episodes)

# replay memory with random sampling
Transition = namedtuple('Transition',
                        ('state', 'discrete_action', 'continuous_action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """save as a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# output Q values for each discrete action index
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


# output continuous actions
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
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #pick discrete action with the largest expected reward
            return pdqn(obs).argmax(dim=1, keepdim=True)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def select_continuous_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return actor(state) + np.random.normal(0, gaussian_exploration_noise)
    else:
        return action_bounds * torch.tensor(np.random.uniform([-1, -1], [1, 1]).reshape(1,-1), dtype=torch.float32)


def update_policy():
    if len(memory) < BATCH_SIZE:
        return (0, 0)
        
    transitions = memory.sample(BATCH_SIZE)
    #convert the batch-array of transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))
    #compute a mask of non-terminal states and concatenate the batch elements
    non_termi_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    next_s_batch = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    d_action_batch = torch.cat(batch.discrete_action)
    c_action_batch = torch.cat(batch.continuous_action)
    reward_batch = torch.cat(batch.reward)

    # compute Q(s_t, k, x_k) based on policy network and action selection for each batch state
    q_values = pdqn(torch.cat([state_batch, c_action_batch], dim=1)).gather(1, d_action_batch)

    # compute expected values based on target network
    max_next_q = torch.zeros((BATCH_SIZE, 1), device=device)
    max_next_q[non_termi_mask] = pdqn_target(torch.cat([next_s_batch, actor_target(next_s_batch)], dim=1)).max(dim=1, keepdim=True)[0].detach()
    expected_q_values = (max_next_q * GAMMA) + reward_batch
     
    # pdqn loss and update pdqn network
    loss_pdqn = F.smooth_l1_loss(q_values, expected_q_values)
    optimizer_pdqn.zero_grad()
    loss_pdqn.backward() # TODO: why we have to use retain_graph=true? #

    # error clipping further improve the stability of the algorithm
    for param in pdqn.parameters():
        param.grad.data.clamp_(-5, 5)
    optimizer_pdqn.step()

    # actor loss and update pdqn network
    loss_actor = -pdqn(torch.cat([state_batch, actor(state_batch)], dim=1)).mean()
    optimizer_actor.zero_grad()
    loss_actor.backward()

    for param in actor.parameters():
        param.grad.data.clamp_(-5, 5)
    optimizer_actor.step()

    #visualize the calculation graph
    #visual_graph = make_dot(loss, params=dict(policy_net.named_parameters()))
    #visual_graph.view()
    return (loss_pdqn.item(), loss_actor.item())



#initialize networks
pdqn = PDQN(state_dim + n_actions, n_actions).to(device)
pdqn_target = PDQN(state_dim + n_actions, n_actions).to(device)
pdqn_target.load_state_dict(pdqn.state_dict())

actor = Actor(state_dim, n_actions, action_bounds)
actor_target = Actor(state_dim, n_actions, action_bounds)
actor_target.load_state_dict(actor.state_dict())

# optimizers
optimizer_pdqn  = optim.Adam(pdqn.parameters(), lr=5e-3)
optimizer_actor = optim.Adam(actor.parameters(), lr=1e-3)

# experience replay
memory = ReplayMemory(20000)

logger = open('./logs/logger.txt', 'w+')


# training process
for epoch in range(num_episodes):
    env = gym.make('nav2d-v0')
    # env initial states TODO: initialize with different position
    initial_position = np.array([env.FIELD_HEIGHT / 2, env.FIELD_WIDTH / 2])
    initial_orientation = -3*np.pi / 4
    #initial_orientation = np.random.normal(loc=0, scale=2*np.pi)
    state = torch.from_numpy(env.reset(initial_position, initial_orientation)).unsqueeze(0).float().to(device)
    done = False

    while not done:
        #select and perform the action
        continuous_action = select_continuous_action(state)
        discrete_action = select_discrete_action(torch.cat([state, continuous_action], dim=1))
        obs, reward, done, _ = env.step((discrete_action.detach().item(), continuous_action.detach().squeeze(0).numpy()))

        if done:
            next_state = None
        else:
            next_state = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        
        episode_rewards[epoch] += reward
        reward = torch.tensor([[reward]], device=device)    #tensor shape(1,1)
        
        #store the transition in memory
        memory.push(state, discrete_action, continuous_action, next_state, reward)
        
        #move to the next state
        state = next_state
        
        #perform one step of optimization
        losses = update_policy()

        #env.render()

    logger.write('{}, {}, {}, {}\n'.format(epoch, episode_rewards[epoch], losses[0], losses[1]))
    logger.flush()

    if epoch % 1000 == 0:
        torch.save(pdqn.state_dict(), './trained_model/trained_pdqn.pt')
        torch.save(actor.state_dict(), './trained_model/trained_actor.pt')

    if episode_rewards[epoch - 20 : epoch + 1].sum() / 100 >= 40:
        torch.save(pdqn.state_dict(), './trained_model/trained_pdqn_good.pt')
        torch.save(actor.state_dict(), './trained_model/trained_actor_good.pt')
        print("Training Finished!\n")

    if epoch % 10 == 0:
        print('epoch:', epoch, 'accumulated reward:', episode_rewards[epoch], 'frames:', steps_done)

    # gradually update target network
    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))
    
    for target_param, param in zip(pdqn_target.parameters(), pdqn.parameters()):
        target_param.data.copy_(param.data * 0.1 * TAU + target_param.data * (1.0 - 0.1 * TAU))

