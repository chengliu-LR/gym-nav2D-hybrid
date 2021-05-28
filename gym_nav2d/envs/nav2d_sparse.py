import gym
gym.logger.set_level(40)
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import numpy as np
from .config import AGENT_CONFIG, ZERO_POINT, FIELD_WIDTH, FIELD_HEIGHT, MIN_SPEED, MAX_SPEED, INERTIA_MOMENT, MAX_POWER, MIN_POWER, GOAL_AREA
from .util import angle_to_pos, norm_angle, norm

# actions
ACTION_LOOKUP = {
    0: 'FORWARD',
    1: 'TURN',
}

# field bounds and turning angles
PARAMETERS_MIN = [
    np.array([MIN_POWER]),
    np.array([-np.pi / 12])
]

PARAMETERS_MAX = [
    np.array([MAX_POWER]),  # width: vertical
    np.array([np.pi / 12])
]

LOW_VECTOR = [
    np.array([ZERO_POINT]), # position x
    np.array([ZERO_POINT]), # position y
    np.array([MIN_SPEED]),  # velocity x
    np.array([MIN_SPEED]),  # velocity y
    np.array([-np.pi])  # orientation
]

HIGH_VECTOR = [
    np.array([FIELD_HEIGHT]),
    np.array([FIELD_WIDTH]),
    np.array([MAX_SPEED]),
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
        self.max_time_step = 50

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.num_actions),
            spaces.Tuple(
                tuple(spaces.Box(PARAMETERS_MIN[i], PARAMETERS_MAX[i], dtype=np.float32) for i in range(self.num_actions))
            )
        ))
        # state = (x_cordinate, y_cordinate, vel_x, vel_y, orientation)
        self.observation_space = spaces.Tuple(
                tuple(spaces.Box(LOW_VECTOR[i], HIGH_VECTOR[i], dtype=np.float32) for i in range(len(LOW_VECTOR)))
        )

        # visualiser
        self.VISUALISER_SCALE_FACTOR = 20
        # env config
        self.FIELD_WIDTH = FIELD_WIDTH
        self.FIELD_HEIGHT = FIELD_HEIGHT

        self.PARAMETERS_MIN = PARAMETERS_MIN
        self.PARAMETERS_MAX = PARAMETERS_MAX

    def step(self, action):
        """
        Action: tuple of (int: discrete_action_index, tuple: params)
        Returns: tuple (ob, reward, end_episode, info)
        """
        act_index = action[0]
        act = ACTION_LOOKUP[act_index]
        param = action[1][act_index] #parameters corresponding to chosen discrete action
        param = np.clip(param, PARAMETERS_MIN[act_index], PARAMETERS_MAX[act_index])

        self.time_step += 1
        if self.time_step == self.max_time_step:
            reward = -self.agent.normalized_goal_distance()
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
            reward = 50.0
        elif not self.agent.in_field():
            end_episode = True
            reward = -20.0 - self.agent.normalized_goal_distance()
        else:
            end_episode = False
            reward = -self.agent.normalized_goal_distance()
        return reward, end_episode

    def reset(self, position, orientation):
        self.agent = Agent(position, orientation)
        return self.get_state() # return state

    def get_state(self):
        """ Return 1-dim numpy array. """
        state = np.concatenate((
            self.agent.position,
            self.agent.velocity,
            [self.agent.orientation]))
        return state

    def render(self, mode='human'):
        self._initialse_window()
        # window, color, position, radius_of_circle
        pygame.draw.circle(self.window, self.__red, self.VISUALISER_SCALE_FACTOR * self.agent.position, 8)
        pygame.display.update()

    def _initialse_window(self):
        # initialise visualiser
        if self.window is None:
            pygame.init()
            width = self._visualiser_scale(self.FIELD_WIDTH)
            height = self._visualiser_scale(self.FIELD_HEIGHT)
            self.window = pygame.display.set_mode((width, height))
            self.__clock = pygame.time.Clock()
            size = (width, height)
            self.__background = pygame.Surface(size)
            self.__white = pygame.Color(255, 255, 255, 0)
            self.__black = pygame.Color(0, 0, 0, 0)
            self.__red = pygame.Color(255, 0, 0, 0)
            self.__background.fill(pygame.Color(0, 125, 0, 0))

            goal_top_left = (self._visualiser_scale(self.FIELD_WIDTH - GOAL_AREA),
                             self._visualiser_scale(self.FIELD_HEIGHT - GOAL_AREA))  # (x, y)
            goal_top_right = (width, self._visualiser_scale(self.FIELD_HEIGHT - GOAL_AREA))
            goal_botton_left = (self._visualiser_scale(self.FIELD_WIDTH - GOAL_AREA), height)

            pygame.draw.line(self.window,
                            self.__white, goal_top_left, goal_top_right, width=5)
            pygame.draw.line(self.window,
                            self.__white, goal_top_left, goal_botton_left, width=5)
    
    def _visualiser_scale(self, value):
        ''' Scale up a value. '''
        return int(self.VISUALISER_SCALE_FACTOR * value)

    ####### for discrete action space ########
    def step_discrete(self, act_index):
        """
        action: (int) discrete_action_index
        Returns: (tuple) (ob, reward, end_episode, info)
        """
        if act_index == 0:
            act = 'FORWARD'
            param = np.array([MAX_POWER])
        elif act_index == 1:
            act = 'TURN'
            param = np.array([np.pi / 12])
        elif act_index == 2:
            act = 'TURN'
            param = np.array([-np.pi / 12])
        else:
            print("action index out of range.")

        self.time_step += 1
        if self.time_step == self.max_time_step:
            reward = -self.agent.normalized_goal_distance()
            end_episode = True
            state = self.get_state()
            return state, reward, end_episode, {}
        
        end_episode = False
        reward, end_episode = self.update(act, param)
        state = self.get_state()

        return state, reward, end_episode, {}


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
        self.velocity = np.clip(self.velocity, -self.speed_max, self.speed_max)
    
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
        self.orientation = norm_angle(orientation)
    
    def turn(self, angle):
        moment = norm_angle(angle)
        speed = norm(self.velocity)
        angle = moment / (1 + INERTIA_MOMENT * speed)
        self.orientation = norm_angle(self.orientation + angle)
    
    def move_forward(self, power):
        power = np.clip(power, MIN_POWER, MAX_POWER)
        self.accelerate(power, self.orientation)
    
    def goal_achieved(self):
        return self.in_area(FIELD_HEIGHT - GOAL_AREA, FIELD_HEIGHT,
                            FIELD_HEIGHT - GOAL_AREA, FIELD_HEIGHT)
    
    def in_field(self):
        return self.in_area(0, FIELD_HEIGHT,
                            0, FIELD_HEIGHT)
    
    def normalized_goal_distance(self):
        return norm(self.position - np.array([FIELD_WIDTH, FIELD_HEIGHT])) / FIELD_HEIGHT