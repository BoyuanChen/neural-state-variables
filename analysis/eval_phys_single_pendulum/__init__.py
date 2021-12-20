from .angle_estimator import obtain_angle
from .physics_estimator import *

import numpy as np


phys_vars_list = {'reject', 'theta', 'vel_theta', 'kinetic energy',
'potential energy', 'total energy'}


def eval_physics(frames):
    num_frames = len(frames)
    reject = np.zeros(num_frames, dtype=bool)
    theta = np.zeros(num_frames)
    # estimate angles
    for p in range(num_frames):
        reject_p, theta_p, _ = obtain_angle(frames[p])
        if reject_p:
            reject[p] = True
            theta[p] = np.nan
        else:
            reject[p] = False
            theta[p] = theta_p
    # calculate velocities
    vel_theta = np.zeros(num_frames)
    sub_ids = np.ma.clump_unmasked(np.ma.masked_array(theta, reject))
    for ids in sub_ids:
        vel_theta[ids] = calc_velocity(theta[ids].copy())
    vel_theta[reject] = np.nan
    # calculate energies
    kinetic_energy, potential_energy, total_energy = calc_energy(theta, vel_theta)
    # save results
    phys = dict.fromkeys(phys_vars_list)
    phys['reject'] = reject
    phys['theta'] = theta
    phys['vel_theta'] = vel_theta
    phys['kinetic energy'] = kinetic_energy
    phys['potential energy'] = potential_energy
    phys['total energy'] = total_energy
    return phys