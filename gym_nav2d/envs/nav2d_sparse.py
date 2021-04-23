############## TODO: add state whether landed ###############

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import numpy as np
from .config import ZERO_POINT, FIELD_WIDTH, FIELD_LENGTH, MIN_SPEED, MAX_SPEED

# actions
ACTION_LOOKUP = {
    0: 'NAV_TO',
    1: 'TURN',
    2: 'LAND',
}

# field bounds
PARAMETERS_MIN = [
    np.array([ZERO_POINT, ZERO_POINT]),
    np.array([-np.pi])
]

PARAMETERS_MAX = [
    np.array([FIELD_LENGTH, FIELD_WIDTH]),
    np.array([np.pi])
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
        self.states = []
        self.window = None

        num_actions = len(ACTION_LOOKUP)
        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),
            spaces.Tuple(
                tuple(spaces.Box(PARAMETERS_MIN[i], PARAMETERS_MAX[i], dtype=np.float64) for i in range(num_actions - 1))
            )
        ))
        # state = (x_cordinate, y_cordinate, speed, orientation)
        self.observation_space = spaces.Tuple(
                tuple(spaces.Box(LOW_VECTOR[i], HIGH_VECTOR[i], dtype=np.float64) for i in range(len(LOW_VECTOR)))
        )

        # visualiser
        self.VISUALISER_SCALE_FACTOR = 20

    def step(self, action):
        # return state, reward, info
        print('This function need to be implemented')
        pass

    def reset(self):
        pass
    
    def render(self, mode='human'):
        self._initialse_window()
        # window, color, position, radius_of_circle
        for i in range(200):
            self.__background.fill(pygame.Color(0, 125, 0, 0))
            pygame.draw.circle(self.window, self.__white, (50+i, 50), 10)
            pygame.display.update()
        return

    def close(self):
        pass

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