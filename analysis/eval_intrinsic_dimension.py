import numpy as np
import os
import sys
from tqdm import tqdm
from PIL import Image
import json
import yaml
import pprint
from munch import munchify
from intrinsic_dimension_estimation import ID_Estimator


def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def remove_duplicates(X):
    return np.unique(X, axis=0)


def eval_id_latent(vars_filepath, if_refine, if_all_methods):
    if if_refine:
        latent = np.load(os.path.join(vars_filepath, 'refine_latent.npy'))
    else:
        latent = np.load(os.path.join(vars_filepath, 'latent.npy'))
    print(f'Number of samples: {latent.shape[0]}; Latent dimension: {latent.shape[1]}')
    latent = remove_duplicates(latent)
    print(f'Number of samples (duplicates removed): {latent.shape[0]}')
    
    estimator = ID_Estimator()
    k_list = (latent.shape[0] * np.linspace(0.008, 0.016, 5)).astype('int')
    print(f'List of numbers of nearest neighbors: {k_list}')
    if if_all_methods:
        dims = estimator.fit_all_methods(latent, k_list)
        np.save(os.path.join(vars_filepath, 'intrinsic_dimension_all_methods.npy'), dims)
    else:
        dims = estimator.fit(latent, k_list)
        np.save(os.path.join(vars_filepath, 'intrinsic_dimension.npy'), dims)


def eval_id_image(data_filepath, test_vid_ids, vars_filepath, if_all_methods):
    print('Reading image data...')
    data = []
    for p_vid_idx in tqdm(test_vid_ids):
        vid_filepath = os.path.join(data_filepath, str(p_vid_idx))
        num_frames = len(os.listdir(vid_filepath))
        suf = os.listdir(vid_filepath)[0].split('.')[-1]
        for p_frame in range(num_frames - 3):
            img_list = []
            for p in range(2):
                img = Image.open(os.path.join(vid_filepath, str(p_frame + p) + '.' + suf))
                img = img.resize((128, 128))
                img = np.array(img) / 255.0
                img_list.append(img)
            data_p = np.concatenate(img_list, 1).reshape([1, -1])
            data.append(data_p)
    data = np.concatenate(data, 0)
    print(f'Number of samples: {data.shape[0]}; Image dimension (flattened): {data.shape[1]}')
    data = remove_duplicates(data)
    print(f'Number of samples (duplicates removed): {data.shape[0]}')

    estimator = ID_Estimator()
    k_list = (data.shape[0] * np.linspace(0.008, 0.016, 5)).astype('int')
    print(f'List of numbers of nearest neighbors: {k_list}')
    if if_all_methods:
        dims = estimator.fit_all_methods(data, k_list)
        np.save(os.path.join(vars_filepath, 'intrinsic_dimension_image_all_methods.npy'), dims)
    else:
        dims = estimator.fit(data, k_list)
        np.save(os.path.join(vars_filepath, 'intrinsic_dimension_image.npy'), dims)


if __name__ == '__main__':
    
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    if_all_methods = (str(sys.argv[3]) == 'all-methods')
    
    if str(sys.argv[2]) == 'model-latent':
        log_dir = '_'.join([cfg.log_dir, cfg.dataset, cfg.model_name, str(cfg.seed)])
        vars_filepath = os.path.join(log_dir, 'variables')
        if_refine = ('refine' in cfg.model_name)
        dims = eval_id_latent(vars_filepath, if_refine, if_all_methods)
    
    elif str(sys.argv[2]) == 'data-image':
        data_filepath = os.path.join(cfg.data_filepath, cfg.dataset)
        with open(os.path.join('../datainfo', cfg.dataset, f'data_split_dict_{cfg.seed}.json'), 'r') as file:
            seq_dict = json.load(file)
        test_vid_ids = seq_dict['test']
        log_dir = '_'.join([cfg.log_dir, cfg.dataset, cfg.model_name, str(cfg.seed)])
        vars_filepath = os.path.join(log_dir, 'variables')
        dims = eval_id_image(data_filepath, test_vid_ids, vars_filepath, if_all_methods)
    
    else:
        assert False, 'Invalid arguments...'
