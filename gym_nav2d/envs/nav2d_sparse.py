############## TODO: add state whether landed ###############

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import numpy as np
from .config import AGENT_CONFIG, ZERO_POINT, FIELD_WIDTH, FIELD_LENGTH, MIN_SPEED, MAX_SPEED, INERTIA_MOMENT, MAX_POWER, MIN_POWER, GOAL_AREA
from .util import angle_to_pos, norm_angle, norm

# actions
ACTION_LOOKUP = {
    0: 'FORWARD',
    1: 'TURN',
}

# field bounds and turning angles
PARAMETERS_MIN = [
    np.array([MAX_POWER]),
    np.array([-np.pi / 36])
]

PARAMETERS_MAX = [
    np.array([MIN_POWER]),  # width: vertical
    np.array([np.pi / 36])
]

LOW_VECTOR = [
    np.array([ZERO_POINT, ZERO_POINT]), # position
    np.array([MIN_SPEED]),  # velocity
    np.array([-np.pi])  # orientation
]

HIGH_VECTOR = [
    np.array([FIELD_LENGTH, FIELD_WIDTH]),
    np.array([MAX_SPEED]),
    np.array([np.pi])
]

class NaviSparseEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.agent = None
        self.states = []
        self.window = None
        self.num_actions = len(ACTION_LOOKUP)

        self.time_step = 0
        self.max_time_step = 200

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.num_actions),
            spaces.Tuple(
                tuple(spaces.Box(PARAMETERS_MIN[i], PARAMETERS_MAX[i], dtype=np.float64) for i in range(self.num_actions))
            )
        ))
        # state = (x_cordinate, y_cordinate, speed, orientation)
        self.observation_space = spaces.Tuple(
                tuple(spaces.Box(LOW_VECTOR[i], HIGH_VECTOR[i], dtype=np.float64) for i in range(len(LOW_VECTOR)))
        )

        # visualiser
        self.VISUALISER_SCALE_FACTOR = 20
        # env config
        self.FIELD_WIDTH = FIELD_WIDTH
        self.FIELD_LENGTH = FIELD_LENGTH

    def step(self, action):
        """
        action: tuple of (int: discrete_action_index, tuple: params)
        Returns: tuple (ob, reward, end_episode, info)
        """
        act_index = action[0]
        act = ACTION_LOOKUP[act_index]
        param = action[1][act_index] # parameters corresponding to choose discrete action
        param = np.clip(param, PARAMETERS_MIN[act_index], PARAMETERS_MAX[act_index])

        self.time_step += 1
        if self.time_step == self.max_time_step:
            reward = -1
            end_episode = True
            state = self.get_state()
            return state, reward, end_episode, {}
        
        end_episode = False
        reward, end_episode = self.update(act, param)
        state = self.get_state()
        return state, reward, end_episode, {}

    def update(self, act, param):
        """ Performs a state transition with the given action, returns the reward and terminal states. """
        self.perform_action(act, param, self.agent)
        self.agent.update_pos_vel()
        return self.terminal_check()

    def perform_action(self, act, param, agent):
        if act == 'FORWARD':
            agent.move_forward(param[0])
        elif act == 'TURN':
            agent.turn(param[0])
        else:
            raise error.InvalidAction("Action not recognized: ", act)

    def terminal_check(self):
        if self.agent.goal_achieved():
            end_episode = True
            reward = 1
        elif not self.agent.in_field():
            end_episode = True
            reward = -1
        else:
            end_episode = False
            reward = 0
        return reward, end_episode

    def reset(self, position, orientation):
        self.agent = Agent(position, orientation)
        return self.get_state(), 0  # return state and reward (0)

    def get_state(self):
        state = np.concatenate((
            self.agent.position,
            self.agent.velocity,
            [self.agent.orientation]))
        return state

    def render(self, mode='human'):
        self._initialse_window()
        # window, color, position, radius_of_circle
        pygame.draw.circle(self.window,
                        self.__red, self.VISUALISER_SCALE_FACTOR * self.agent.position,
                        10)
    
        pygame.display.update()

    def _initialse_window(self):
        # initialise visualiser
        if self.window is None:
            pygame.init()
            width = self.__visualiser_scale(FIELD_WIDTH)
            height = self.__visualiser_scale(FIELD_LENGTH)
            self.window = pygame.display.set_mode((width, height))
            self.__clock = pygame.time.Clock()
            size = (width, height)
            self.__background = pygame.Surface(size)
            self.__white = pygame.Color(255, 255, 255, 0)
            self.__black = pygame.Color(0, 0, 0, 0)
            self.__red = pygame.Color(255, 0, 0, 0)
            self.__background.fill(pygame.Color(0, 125, 0, 0))
    
    def __visualiser_scale(self, value):
        ''' Scale up a value. '''
        return int(self.VISUALISER_SCALE_FACTOR * value)

class Entity():
    """ Base entity class, representing moving objects like quads or other types of drones. """

    def __init__(self, config):
        self.accel_max = config['ACCEL_MAX']
        self.speed_max = config['SPEED_MAX']
        # what is power rate?
        self.power_rate = config['POWER_RATE']
        self.decay = config['DECAY']    # velocity decay
        self.position = np.array([0., 0.])
        self.velocity = np.array([0., 0.])
    
    def update_pos_vel(self):
        self.position += self.velocity
        self.velocity *= self.decay
    
    def accelerate(self, power, theta):
        """ Applies a power to the entity in direction theta. """
        acceleration = self.power_rate * float(power) * angle_to_pos(theta)
        acceleration = np.clip(acceleration, -self.accel_max, self.accel_max)
        self.velocity += acceleration
        self.velocity = np.clip(self.velocity, -self.speed_max, self.accel_max)
    
    def in_area(self, left, right, bottom, top):
        xval, yval = self.position
        in_horizontal = left <= xval <= right
        in_vertical = bottom <= yval <= top
        return in_horizontal and in_vertical

class Agent(Entity):
    def __init__(self, position, orientation):
        #super(Agent, self).__init__()
        Entity.__init__(self, AGENT_CONFIG)
        self.position = position
        self.orientation = orientation
    
    def turn(self, angle):
        moment = norm_angle(angle)
        speed = norm(self.velocity)
        angle = moment / (1 + INERTIA_MOMENT * speed)
        self.orientation = self.orientation + angle
    
    def move_forward(self, power):
        power = np.clip(power, MIN_POWER, MAX_POWER)
        self.accelerate(power, self.orientation)
    
    def goal_achieved(self):
        return self.in_area(FIELD_LENGTH - GOAL_AREA, FIELD_LENGTH,
                            FIELD_LENGTH - GOAL_AREA, FIELD_LENGTH)
    
    def in_field(self):
        return self.in_area(0, FIELD_LENGTH,
                            0, FIELD_LENGTH)