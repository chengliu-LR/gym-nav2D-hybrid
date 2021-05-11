import math
import numpy as np

def angle_to_pos(theta):
    """ Compute the position on a unit circle at angle theta. """
    return np.array([np.cos(theta), np.sin(theta)])

def norm(vec2d):
    # from numpy.linalg import norm
    # faster to use custom norm because we know the vectors are always 2D
    assert len(vec2d) == 2
    return math.sqrt(vec2d[0]*vec2d[0] + vec2d[1]*vec2d[1])

def norm_angle(angle):
    """ Normalize the angle between -pi and pi. """
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle
