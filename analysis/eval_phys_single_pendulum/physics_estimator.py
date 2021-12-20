'''
This script provides utility functions estimating angular velocities
and other physical quantities from angles of single pendulum.
'''

import numpy as np
from scipy.interpolate import CubicSpline

# physical parameters
fps = 60  # frames per second
l = 0.5   # pendulum rod length (m)
m = 1.0   # bob mass (kg)
g = 9.81  # gravitational acceleration (m/s^2)


'''
Calculate the absolute difference between two angles th1 and th2 on a circle.
Assumed that the absolute difference between the two angles is within range (0,pi).
'''
def calc_diff(th1, th2):
    diff = np.abs(th2 - th1)
    diff = np.minimum(diff, 2*np.pi-diff)
    return diff

'''
Calculate the average of two angles th1 and th2 on a circle.
Assumed that the absolute difference between the two angles is within range (0,pi).
'''
def calc_avrg(th1, th2):
    avrg = (th1 + th2) / 2
    diff = np.abs(th2 - th1)
    if diff > np.pi:
        avrg -= np.pi
    if avrg < 0:
        avrg += 2*np.pi
    return avrg 

'''
Normalize the angle to (-pi, pi).
'''
def normalize_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

'''
Calculate angular velocities from a sequence of angles
using numerical differentiation.
method='fd': finite difference;
method='spline': cubic spline fitting.
'''
def calc_velocity(th, method='spline'):
    len_seq = th.shape[0]

    # isolated data
    if len_seq == 1:
        return np.nan
    
    # preprocessing: periodic extension of angles
    for i in range(1, len_seq):
        if th[i] - th[i-1] > np.pi:
            th[i:] -= 2*np.pi
        elif th[i] - th[i-1] < -np.pi:
            th[i:] += 2*np.pi
    
    vel_th = np.zeros(len_seq)

    # finite difference
    if method == 'fd':
        for i in range(1, len_seq):
            vel_th[i] = (th[i] - th[i-1]) * fps
        vel_th[0] = (th[1] - th[0]) * fps
    
    # cubic spline fitting
    elif method == 'spline':
        t = np.arange(len_seq) / fps
        cs = CubicSpline(t, th)
        vel_th = cs(t, 1)
        # use finite difference at boundary points to improve accuracy
        vel_th[0] = (th[1] - th[0]) * fps
        vel_th[-1] = (th[-1] - th[-2]) * fps
    
    else:
        assert False, 'Unrecognizable differentiation method!'
    
    return vel_th

'''
Calculate energies from angles and angular velocities
Moment of inertia of the pendulum: I=ml^2/3
'''
def calc_energy(th, vel_th):
    T = m * l**2 / 6 * vel_th**2
    V = - m * g * l / 2 * np.cos(th)
    E = T + V
    return T, V, E
