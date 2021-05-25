import numpy as np

ZERO_POINT = 0.0
FIELD_LENGTH = 15
FIELD_WIDTH = 15

GOAL_AREA = 5

MAX_POWER = 100
MIN_POWER = -100

MIN_SPEED = -5
MAX_SPEED = 5

INERTIA_MOMENT = 1

AGENT_CONFIG = {
    'POWER_RATE': 0.006,    # rate parameter multiplied to the real power
    'SIZE': 0.9,
    'RAND': 0.1,
    'ACCEL_MAX': 1.0,
    'SPEED_MAX': 1.0,
    'DECAY': 0.2,
    'MASS': 60
}