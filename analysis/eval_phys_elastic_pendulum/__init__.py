from .angle_estimator import obtain_angle_stretch
from .physics_estimator import *

import numpy as np


phys_vars_list = {'reject', 'theta_1', 'vel_theta_1', 'theta_2', 'vel_theta_2', 'z', 'vel_z',
'kinetic energy', 'potential energy', 'total energy'}


def eval_physics(frames):
    num_frames = len(frames)
    reject = np.zeros(num_frames, dtype=bool)
    theta_1 = np.zeros(num_frames)
    theta_2 = np.zeros(num_frames)
    z = np.zeros(num_frames)
    # estimate angles and stretch
    for p in range(num_frames):
        reject_p, angles_p, stretch_p, _ = obtain_angle_stretch(frames[p])
        if reject_p:
            reject[p] = True
            theta_1[p] = np.nan
            theta_2[p] = np.nan
            z[p] = np.nan
        else:
            reject[p] = False
            theta_1[p] = angles_p[0]
            theta_2[p] = angles_p[1]
            z[p] = stretch_p
    # calculate velocities
    vel_theta_1 = np.zeros(num_frames)
    vel_theta_2 = np.zeros(num_frames)
    vel_z = np.zeros(num_frames)
    sub_ids = np.ma.clump_unmasked(np.ma.masked_array(theta_1, reject))
    for ids in sub_ids:
        vel_theta_1[ids] = calc_velocity(theta_1[ids].copy())
        vel_theta_2[ids] = calc_velocity(theta_2[ids].copy())
        vel_z[ids] = calc_velocity(z[ids].copy(), periodic=False)
    vel_theta_1[reject] = np.nan
    vel_theta_2[reject] = np.nan
    vel_z[reject] = np.nan
    # calculate energies
    kinetic_energy, potential_energy, total_energy = calc_energy(theta_1, theta_2, z, vel_theta_1, vel_theta_2, vel_z)
    # save results
    phys = dict.fromkeys(phys_vars_list)
    phys['reject'] = reject
    phys['theta_1'] = theta_1
    phys['vel_theta_1'] = vel_theta_1
    phys['theta_2'] = theta_2
    phys['vel_theta_2'] = vel_theta_2
    phys['z'] = z
    phys['vel_z'] = vel_z
    phys['kinetic energy'] = kinetic_energy
    phys['potential energy'] = potential_energy
    phys['total energy'] = total_energy
    return phys