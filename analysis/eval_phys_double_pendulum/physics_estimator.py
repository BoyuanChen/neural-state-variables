'''
This script provides utility functions estimating angular velocities
and other physical quantities from angles of double pendulum.
'''

import numpy as np
from scipy.interpolate import CubicSpline

# physical parameters
fps = 60 # frames per second
L1, L2 = 0.205, 0.179  # pendulum rod lengths (m)
w1, w2 = 0.038, 0.038  # pendulum rod widths (m)
m1, m2 = 0.262, 0.110  # bob masses (kg)
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
'''
def calc_energy(th1, th2, vel_th1, vel_th2):
    # centers of masses in x-y coordinates
    x1 = (L1 / 2) * np.sin(th1)
    y1 = (-L1 / 2) * np.cos(th1)

    x2 = (L1 * np.sin(th1)) + (L2 / 2) * np.sin(th2)
    y2 = (-L1 * np.cos(th1)) - (L2 / 2) * np.cos(th2)

    # velocities in x-y coordinates
    vel_x1 = vel_th1 * (L1 / 2) * np.cos(th1)
    vel_y1 = vel_th1 * (L1 / 2) * np.sin(th1)

    vel_x2 = vel_th1 * L1 * np.cos(th1) + (vel_th2 / 2) * L2 * np.cos(th2)
    vel_y2 = vel_th1 * L1 * np.sin(th1) + (vel_th2 / 2) * L2 * np.sin(th2)

    # moments of inertia
    I1 = (1. / 12.) * m1 * (w1 ** 2 + L1 ** 2)
    I2 = (1. / 12.) * m2 * (w2 ** 2 + L2 ** 2)

    # potential energy
    V = g * (m1 * y1 + m2 * y2)
    # kinetic energy (translation + rotation)
    T = 0.5 * m1 * (vel_x1 ** 2 + vel_y1 ** 2) + 0.5 * m2 * (vel_x2 ** 2 + vel_y2 ** 2) + 0.5 * I1 * (
                vel_th1 ** 2) + 0.5 * I2 * (vel_th2 ** 2)
    # total energy
    E = T + V

    return T, V, E