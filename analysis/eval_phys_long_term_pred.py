"""
Evaluate long-term prediction accuracy in physical variables.
"""

import os
import sys
import numpy as np
from scipy import stats
import cv2
import json
import yaml
import pprint
from tqdm import tqdm
from munch import munchify


'''
Calculate the absolute difference between two angles th1 and th2 on a circle.
Assumed that the absolute difference between the two angles is within range (0,pi).
'''
def calc_diff(th1, th2):
    diff = np.abs(th2 - th1)
    diff = np.minimum(diff, 2*np.pi-diff)
    return diff

'''
Calculate the pixel mean square error between two images.
'''
def calc_pixel_MSE(img1, img2):
    return np.mean((img1/255.0 - img2/255.0) ** 2)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)


def eval_pred_physics(data_filepath, pred_save_path, vid_ids, phys_vars_list, eval_physics):
    phys = {p_var:[] for p_var in phys_vars_list}

    for idx in tqdm(vid_ids):
        data_vid_filepath = os.path.join(data_filepath, str(idx))
        pred_vid_filepath = os.path.join(pred_save_path, str(idx))
        num_frames = len(os.listdir(data_vid_filepath))
        num_pred_frames = len(os.listdir(pred_vid_filepath))
        frames = []

        for p in range(num_frames):
            if (num_pred_frames == num_frames - 2) and (p == 0 or p == 1):
                # read the initial two frames from ground truth data
                frame_p = cv2.imread(os.path.join(data_vid_filepath, str(p)+'.png'))
            else:
                frame_p = cv2.imread(os.path.join(pred_vid_filepath, str(p)+'.png'))
            frames.append(frame_p)
        phys_tmp = eval_physics(frames)
        
        for p_var in phys_vars_list:
            phys[p_var].append(phys_tmp[p_var])

    for p_var in phys_vars_list:
        phys[p_var] = np.array(phys[p_var])

    return phys


def load_data_physics(data_filepath, vid_ids, phys_vars_list):
    phys = np.load(os.path.join(data_filepath, 'phys_vars.npy'), allow_pickle=True).item()
    for p_var in phys_vars_list:
        phys[p_var] = phys[p_var][vid_ids]
    return phys


def eval_physics_error(phys_pred, phys_data, phys_vars_list):
    phys_vars_list_2 = [p_var for p_var in phys_vars_list if p_var!='reject']
    phys_error = {}

    phys_error['reject'] = phys_pred['reject'].copy()
    if 'reject' in phys_data.keys():
        phys_error['reject_data'] = phys_data['reject'].copy()
    else:
        phys_error['reject_data'] = np.zeros(phys_pred['reject'].shape)

    for p_var in phys_vars_list_2:
        if p_var in ['theta', 'theta_1', 'theta_2']:
            phys_error[p_var] = calc_diff(phys_pred[p_var], phys_data[p_var])
        else:
            phys_error[p_var] = np.abs(phys_pred[p_var] - phys_data[p_var])

    return phys_error


def eval_pixel_error(data_filepath, pred_save_path, vid_ids):
    pixel_error = []

    for idx in tqdm(vid_ids):
        data_vid_filepath = os.path.join(data_filepath, str(idx))
        pred_vid_filepath = os.path.join(pred_save_path, str(idx))
        num_frames = len(os.listdir(data_vid_filepath))
        pixel_error_idx = []

        for p in range(num_frames):
            if p == 0 or p == 1:
                pixel_error_idx.append(0)
            else:
                data = cv2.imread(os.path.join(data_vid_filepath, str(p)+'.png'))
                pred = cv2.imread(os.path.join(pred_vid_filepath, str(p)+'.png'))
                pixel_error_idx.append(calc_pixel_MSE(data, pred))
        
        pixel_error.append(pixel_error_idx)

    return np.array(pixel_error)


def main():
    config_filepath = str(sys.argv[1])
    pred_save_path = str(sys.argv[2])

    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)

    data_filepath = os.path.join(cfg.data_filepath, cfg.dataset)
    with open(os.path.join('../datainfo', cfg.dataset, f'data_split_dict_{cfg.seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    vid_ids = seq_dict['test']

    if cfg.dataset == 'single_pendulum':
        from eval_phys_single_pendulum import phys_vars_list, eval_physics
    elif cfg.dataset == 'double_pendulum':
        from eval_phys_double_pendulum import phys_vars_list, eval_physics  
    elif cfg.dataset == 'elastic_pendulum':
        from eval_phys_elastic_pendulum import phys_vars_list, eval_physics
    else:
        assert False, 'Unknown system...'

    phys_pred = eval_pred_physics(data_filepath, pred_save_path, vid_ids, phys_vars_list, eval_physics)
    phys_data = load_data_physics(data_filepath, vid_ids, phys_vars_list)

    phys_error = eval_physics_error(phys_pred, phys_data, phys_vars_list)
    pixel_error = eval_pixel_error(data_filepath, pred_save_path, vid_ids)

    np.save(os.path.join(pred_save_path, 'phys_vars.npy'), phys_pred)
    np.save(os.path.join(pred_save_path, 'phys_error.npy'), phys_error)
    np.save(os.path.join(pred_save_path, 'pixel_error.npy'), pixel_error)


if __name__ == '__main__':    
    main()