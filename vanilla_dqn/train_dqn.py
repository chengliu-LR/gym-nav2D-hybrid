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
num_episodes = 5000
episode_rewards = np.zeros(num_episodes)

#Replay Memory. It stores the transitions that the agent observes, allowing reusing this data later.
#By sampling from it randomly, the transitions that build up a batch are decorrelated.
#It greatly stabilizes and improves the DQN training procedure.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #pick action with the largest expected reward
            return policy_net(state).argmax(dim=1, keepdim=True)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def update_policy():
    if len(memory) < BATCH_SIZE:
        return
        
    transitions = memory.sample(BATCH_SIZE)
    #convert the batch-array of transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))
    #compute a mask of non-terminal states and concatenate the batch elements
    non_termi_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    concatenate_next_state = [s for s in batch.next_state if s is not None]
    next_state_batch = torch.cat(concatenate_next_state)
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    #compute Q(s_t, a) based on policy network and action selection for each batch state
    q_values = policy_net(state_batch).gather(1, action_batch)
    #compute expected values based on target network
    max_next_q = torch.zeros((BATCH_SIZE, 1), device=device)
    max_next_q[non_termi_mask] = target_net(next_state_batch).max(dim=1,
                                                                keepdim=True)[0].detach()
    expected_q_values = (max_next_q * GAMMA) + reward_batch
    #compute loss and optimizer update
    loss = F.smooth_l1_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    #error clipping further improve the stability of the algorithm
    for param in policy_net.parameters():
        param.grad.data.clamp_(-2, 2)
    optimizer.step()

    #visualize the calculation graph
    #visual_graph = make_dot(loss, params=dict(policy_net.named_parameters()))
    #visual_graph.view()

#initialize the networks, optimizers and the memory
policy_net = FcDQN(state_dim, n_actions).to(device)
target_net = FcDQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
#target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

memory = ReplayMemory(5000)

logger = open('./logs/reward_logger.txt', 'w+')

# training process
for epoch in range(num_episodes):
    env = gym.make('nav2d-v0')
    # env initial states
    initial_position = np.array([env.FIELD_LENGTH / 2, env.FIELD_WIDTH / 2])
    state = torch.from_numpy(env.reset(initial_position, np.random.normal(loc=0, scale=2*np.pi))).unsqueeze(0).float().to(device)
    done = False
    #until the game is end
    while not done:
        #select and perform the action
        action = select_action(state)
        obs, reward, done, _ = env.step_discrete(action.item())
        if done:
            next_state = None
        else:
            next_state = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        episode_rewards[epoch] += reward
        reward = torch.tensor([[reward]], device=device)    #tensor shape(1,1)
        #store the transition in memory
        memory.push(state, action, next_state, reward)
        #move to the next state
        state = next_state
        #perform one step of optimization
        update_policy()
        #env.render()

    logger.write('{}, {}\n'.format(epoch, episode_rewards[epoch]))
    logger.flush()

    if epoch % 500 == 0:
        torch.save(policy_net.state_dict(), './trained_model/nav2d-trained-model.pt')

    if episode_rewards[epoch - 20 : epoch + 1].sum() / 20 > 40:
        torch.save(policy_net.state_dict(), './trained_model/nav2d-trained-model-good.pt')
        print("Training Finished!\n")

    #if epoch % 1 == 0:
    print('epoch:', epoch, 'accumulated reward:', episode_rewards[epoch], 'frames:', steps_done)

    if epoch % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

