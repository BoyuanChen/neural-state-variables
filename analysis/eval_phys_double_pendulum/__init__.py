from .angle_estimator import obtain_angle
from .physics_estimator import *

import numpy as np


phys_vars_list = {'reject', 'theta_1', 'vel_theta_1', 'theta_2', 'vel_theta_2', 'kinetic energy',
'potential energy', 'total energy'}


def eval_physics(frames):
    num_frames = len(frames)
    reject = np.zeros(num_frames, dtype=bool)
    theta_1 = np.zeros(num_frames)
    theta_2 = np.zeros(num_frames)
    # estimate angles
    for p in range(num_frames):
        reject_p, angles_p, _ = obtain_angle(frames[p])
        if reject_p:
            reject[p] = True
            theta_1[p] = np.nan
            theta_2[p] = np.nan
        else:
            reject[p] = False
            theta_1[p] = angles_p[0]
            theta_2[p] = angles_p[1]
    # calculate velocities
    vel_theta_1 = np.zeros(num_frames)
    vel_theta_2 = np.zeros(num_frames)
    sub_ids = np.ma.clump_unmasked(np.ma.masked_array(theta_1, reject))
    for ids in sub_ids:
        vel_theta_1[ids] = calc_velocity(theta_1[ids].copy())
        vel_theta_2[ids] = calc_velocity(theta_2[ids].copy())
    vel_theta_1[reject] = np.nan
    vel_theta_2[reject] = np.nan
    # calculate energies
    kinetic_energy, potential_energy, total_energy = calc_energy(theta_1, theta_2, vel_theta_1, vel_theta_2)
    # save results
    phys = dict.fromkeys(phys_vars_list)
    phys['reject'] = reject
    phys['theta_1'] = theta_1
    phys['vel_theta_1'] = vel_theta_1
    phys['theta_2'] = theta_2
    phys['vel_theta_2'] = vel_theta_2
    phys['kinetic energy'] = kinetic_energy
    phys['potential energy'] = potential_energy
    phys['total energy'] = total_energy
    return phys