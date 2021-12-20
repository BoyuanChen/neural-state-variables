import os
import sys
import cv2
import numpy as np
from tqdm import tqdm


def eval_phys_data_single_pendulum(data_filepath, num_vids, num_frms, save_path):
    from eval_phys_single_pendulum import eval_physics, phys_vars_list
    phys = {p_var:[] for p_var in phys_vars_list}

    for n in tqdm(range(num_vids)):
        seq_filepath = os.path.join(data_filepath, str(n))
        frames = []
        for p in range(num_frms):
            frame_p = cv2.imread(os.path.join(seq_filepath, str(p)+'.png'))
            frames.append(frame_p)
        phys_tmp = eval_physics(frames)
        for p_var in phys_vars_list:
            phys[p_var].append(phys_tmp[p_var])
    for p_var in phys_vars_list:
        phys[p_var] = np.array(phys[p_var])

    np.save(save_path, phys)


def eval_phys_data_double_pendulum(data_filepath, num_vids, num_frms, save_path):
    from eval_phys_double_pendulum import eval_physics, phys_vars_list
    phys = {p_var:[] for p_var in phys_vars_list}

    for n in tqdm(range(num_vids)):
        seq_filepath = os.path.join(data_filepath, str(n))
        frames = []
        for p in range(num_frms):
            frame_p = cv2.imread(os.path.join(seq_filepath, str(p)+'.png'))
            frames.append(frame_p)
        phys_tmp = eval_physics(frames)
        for p_var in phys_vars_list:
            phys[p_var].append(phys_tmp[p_var])
    for p_var in phys_vars_list:
        phys[p_var] = np.array(phys[p_var])

    # remove outliers
    thresh_1 = np.nanpercentile(np.abs(phys['vel_theta_1']), 98)
    thresh_2 = np.nanpercentile(np.abs(phys['vel_theta_2']), 98)
    for n in range(num_vids):
        for p in range(num_frms):
            if (not np.isnan(phys['vel_theta_1'][n, p]) and np.abs(phys['vel_theta_1'][n, p]) >= thresh_1) \
            or (not np.isnan(phys['vel_theta_2'][n, p]) and np.abs(phys['vel_theta_2'][n, p]) >= thresh_2):
                phys['vel_theta_1'][n, p] = np.nan
                phys['vel_theta_2'][n, p] = np.nan
                phys['kinetic energy'][n, p] = np.nan
                phys['total energy'][n, p] = np.nan

    np.save(save_path, phys)


def eval_phys_data_elastic_pendulum(data_filepath, num_vids, num_frms, save_path):
    from eval_phys_elastic_pendulum import eval_physics, phys_vars_list
    phys = {p_var:[] for p_var in phys_vars_list}

    for n in tqdm(range(num_vids)):
        seq_filepath = os.path.join(data_filepath, str(n))
        frames = []
        for p in range(num_frms):
            frame_p = cv2.imread(os.path.join(seq_filepath, str(p)+'.png'))
            frames.append(frame_p)
        phys_tmp = eval_physics(frames)
        for p_var in phys_vars_list:
            phys[p_var].append(phys_tmp[p_var])
    for p_var in phys_vars_list:
        phys[p_var] = np.array(phys[p_var])

    # remove outliers
    thresh_1 = np.nanpercentile(np.abs(phys['vel_theta_1']), 98)
    thresh_2 = np.nanpercentile(np.abs(phys['vel_theta_2']), 98)
    thresh_z = np.nanpercentile(np.abs(phys['vel_z']), 98)
    for n in range(num_vids):
        for p in range(num_frms):
            if (not np.isnan(phys['vel_theta_1'][n, p]) and np.abs(phys['vel_theta_1'][n, p]) >= thresh_1) \
            or (not np.isnan(phys['vel_theta_2'][n, p]) and np.abs(phys['vel_theta_2'][n, p]) >= thresh_2) \
            or (not np.isnan(phys['vel_z'][n, p]) and np.abs(phys['vel_z'][n, p]) >= thresh_z):
                phys['vel_theta_1'][n, p] = np.nan
                phys['vel_theta_2'][n, p] = np.nan
                phys['vel_z'][n, p] = np.nan
                phys['kinetic energy'][n, p] = np.nan
                phys['total energy'][n, p] = np.nan

    np.save(save_path, phys)


if __name__ == '__main__':
    dataset = str(sys.argv[1])
    data_filepath = str(sys.argv[2])
    save_path = os.path.join(data_filepath, 'phys_vars.npy')
    
    if dataset == 'single_pendulum':
        eval_phys_data_single_pendulum(data_filepath, 1200, 60, save_path)
    elif dataset == 'double_pendulum':
        eval_phys_data_double_pendulum(data_filepath, 1100, 60, save_path)
    elif dataset == 'elastic_pendulum':
        eval_phys_data_elastic_pendulum(data_filepath, 1200, 60, save_path)
    else:
        assert False, 'Unknown system...'