'''
This script provides utility functions estimating velocities
and other physical quantities of elastic double pendulum.
'''

import numpy as np
from scipy.interpolate import CubicSpline

# physical parameters
fps = 60     # frames per second
L_0 = 0.205  # elastic pendulum unstretched length (m)
L = 0.179    # rigid pendulum rod length (m)
w = 0.038    # rigid pendulum rod width (m)
m = 0.110    # rigid pendulum mass (kg)
g = 9.81     # gravitational acceleration (m/s^2)
k = 40.0     # elastic constant (kg/s^2)

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
Calculate velocities from a sequence of data
using numerical differentiation.
method='fd': finite difference;
method='spline': cubic spline fitting.
'''
def calc_velocity(x, method='spline', periodic=True):
    len_seq = x.shape[0]

    # isolated data
    if len_seq == 1:
        return np.nan
    
    # preprocessing: periodic extension of angles
    if periodic:
        for i in range(1, len_seq):
            if x[i] - x[i-1] > np.pi:
                x[i:] -= 2*np.pi
            elif x[i] - x[i-1] < -np.pi:
                x[i:] += 2*np.pi
    
    vel_x = np.zeros(len_seq)

    # finite difference
    if method == 'fd':
        for i in range(1, len_seq):
            vel_x[i] = (x[i] - x[i-1]) * fps
        vel_x[0] = (x[1] - x[0]) * fps
    
    # cubic spline fitting
    elif method == 'spline':
        t = np.arange(len_seq) / fps
        cs = CubicSpline(t, x)
        vel_x = cs(t, 1)
        # use finite difference at boundary points to improve accuracy
        vel_x[0] = (x[1] - x[0]) * fps
        vel_x[-1] = (x[-1] - x[-2]) * fps
    
    else:
        assert False, 'Unrecognizable differentiation method!'
    
    return vel_x

'''
Calculate energies
'''
def calc_energy(th1, th2, z, vel_th1, vel_th2, vel_z):
    # center of mass in x-y coordinates
    x = (L_0 + z) * np.sin(th1) + (L / 2) * np.sin(th2)
    y = -(L_0 + z) * np.cos(th1) - (L / 2) * np.cos(th2)

    # velocities in x-y coordinates
    vel_x = vel_th1 * (L_0 + z) * np.cos(th1) + vel_th2 * (L / 2) * np.cos(th2) + vel_z * np.sin(th1)
    vel_y = vel_th1 * (L_0 + z) * np.sin(th1) + vel_th2 * (L / 2) * np.sin(th2) - vel_z * np.cos(th1)

    # moment of inertia
    I = (1. / 12.) * m * (w ** 2 + L ** 2)

    # potential energy (gravitational + elastic)
    V = m * g * y + 0.5 * k * z**2

    # kinetic energy (translation + rotation)
    T = 0.5 * m * (vel_x ** 2 + vel_y ** 2) + 0.5 * I * vel_th2 ** 2
    
    # total energy
    E = T + V

    return T, V, E