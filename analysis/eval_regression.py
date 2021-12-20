import os
import sys
import numpy as np
from tqdm import tqdm
import json
import yaml
import pprint
from munch import munchify
from latent_regression import pca, mlp_regress


def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)


# parse the video no. and frame no. from the data id (example : '7_0.png')
def parse_data_id(data_id):
    lhs, rhs = data_id.split('_')
    vid_n = int(lhs)
    frm_n = int(rhs.split('.')[0])
    return vid_n, frm_n


def physical_variables_from_data_ids(phys_all, ids):
    phys_vars_list = []
    for p_var in phys_all.keys():
        if p_var == 'reject':
            continue
        for t in range(4):
            phys_vars_list.append(f'{p_var} (t={t})')
    
    num_data = ids.shape[0]
    phys = {p_var:np.zeros(num_data) for p_var in phys_vars_list}
    
    for n in range(num_data):
        vid_n, frm_n = parse_data_id(ids[n])
        for p_var in phys_all.keys():
            if p_var == 'reject':
                continue
            for t in range(4):
                phys[f'{p_var} (t={t})'][n] = phys_all[p_var][vid_n, frm_n+t]
    
    return phys


def main():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)

    if_refine = ('refine' in cfg.model_name)
    phys_all = np.load(os.path.join(cfg.data_filepath, cfg.dataset, 'phys_vars.npy'), allow_pickle=True).item()
    log_dir = '_'.join([cfg.log_dir, cfg.dataset, cfg.model_name, str(cfg.seed)])

    # latent vectors and physical variables (train and test)
    ids_train = np.load(os.path.join(log_dir, 'variables_train', 'ids.npy'))
    ids_test = np.load(os.path.join(log_dir, 'variables', 'ids.npy'))
    if if_refine:
        latent_train = np.load(os.path.join(log_dir, 'variables_train', 'refine_latent.npy'))
        latent_test = np.load(os.path.join(log_dir, 'variables', 'refine_latent.npy'))
    else:
        latent_train = np.load(os.path.join(log_dir, 'variables_train', 'latent.npy'))
        latent_test = np.load(os.path.join(log_dir, 'variables', 'latent.npy'))
    phys_train = physical_variables_from_data_ids(phys_all, ids_train)
    phys_test = physical_variables_from_data_ids(phys_all, ids_test)
    phys_vars_list = phys_train.keys()

    # remove outliers
    is_outlier_train = np.full(latent_train.shape[0], False)
    for p_var in phys_vars_list:
        is_outlier_train = is_outlier_train | np.isnan(phys_train[p_var])
    for p_var in phys_vars_list:
        phys_train[p_var] = phys_train[p_var][~is_outlier_train]
    latent_train = latent_train[~is_outlier_train]
    is_outlier_test = np.full(latent_test.shape[0], False)
    for p_var in phys_vars_list:
        is_outlier_test = is_outlier_test | np.isnan(phys_test[p_var])
    for p_var in phys_vars_list:
        phys_test[p_var] = phys_test[p_var][~is_outlier_test]
    latent_test = latent_test[~is_outlier_test]
    print(f'Valid training samples: {latent_train.shape[0]}; test samples: {latent_test.shape[0]}.')

    # only use first few principal components (optional)
    if str(sys.argv[2]) == 'NA': 
        num_components = None
    else:
        num_components = int(sys.argv[2])
        latent_train, latent_test, pca_model = pca(latent_train, latent_test, num_components, cfg.seed)
    
    num_samples = 3
    sample_size = int(0.3 * latent_train.shape[0])
       
    train_error = {p_var:np.zeros(num_samples) for p_var in phys_vars_list}
    test_error = {p_var:np.zeros(num_samples) for p_var in phys_vars_list}
    reg_models = {p_var:np.empty(num_samples, dtype=object) for p_var in phys_vars_list}

    rng = np.random.default_rng(cfg.seed)
    for i in range(num_samples):
        print('random samples #'+str(i+1))
        samp_ids = rng.choice(latent_train.shape[0], sample_size, replace=False)
        for p_var in phys_vars_list:
            print('doing '+p_var+' regression...')
            train_error[p_var][i], test_error[p_var][i], reg_models[p_var][i] = mlp_regress(latent_train[samp_ids], latent_test, phys_train[p_var][samp_ids], phys_test[p_var], cfg.seed)

    if num_components == None:
        np.save(os.path.join(log_dir, 'variables', 'regression_results.npy'), [train_error, test_error, reg_models])
    else:
        np.save(os.path.join(log_dir, 'variables', 'regression_results_pca_'+str(num_components)+'.npy'), [train_error, test_error, pca_model, reg_models])


if __name__ =='__main__':
    main()