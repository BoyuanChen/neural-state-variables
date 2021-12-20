import os
import sys
import glob
import yaml
import json
import torch
import pprint
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from munch import munchify
from models_latentpred import VisLatentDynamicsModel
from dataset import NeuralPhysDataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)


def get_data(filepath):
    data = Image.open(filepath)
    data = data.resize((128, 128))
    data = np.array(data)
    data = torch.tensor(data / 255.0)
    data = data.permute(2, 0, 1).float()
    return data


def main():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    high_dim_checkpoint_filepath = str(sys.argv[3])
    refine_checkpoint_filepath = str(sys.argv[4])
    pred_save_path = str(sys.argv[5])
    pred_len = int(sys.argv[6])
    
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = VisLatentDynamicsModel(lr=cfg.lr,
                                   seed=cfg.seed,
                                   if_cuda=cfg.if_cuda,
                                   if_test=True,
                                   gamma=cfg.gamma,
                                   log_dir=log_dir,
                                   train_batch=cfg.train_batch,
                                   val_batch=cfg.val_batch,
                                   test_batch=cfg.test_batch,
                                   num_workers=cfg.num_workers,
                                   model_name=cfg.model_name,
                                   data_filepath=cfg.data_filepath,
                                   dataset=cfg.dataset,
                                   lr_schedule=cfg.lr_schedule)

    model.load_model(checkpoint_filepath)
    model.load_high_dim_refine_model(high_dim_checkpoint_filepath, refine_checkpoint_filepath)
    model.extract_decoder_from_refine_model()
    model = model.to('cuda')
    model.eval()
    model.freeze()

    # get all the test video ids
    data_filepath_base = os.path.join(cfg.data_filepath, cfg.dataset)
    with open(os.path.join('../datainfo', cfg.dataset, f'data_split_dict_{cfg.seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    test_vid_ids = seq_dict['test']
    
    errors = []
    for p_vid_idx in tqdm(test_vid_ids):
        data_vid_filepath = os.path.join(data_filepath_base, str(p_vid_idx))
        pred_vid_filepath = os.path.join(pred_save_path, str(p_vid_idx))
        total_num_frames = len(os.listdir(data_vid_filepath))
        suf = os.listdir(data_vid_filepath)[0].split('.')[-1]
        
        error_p = []
        for start_frame_idx in range(total_num_frames - 3):
            if start_frame_idx % 2 != 0:
                continue
            # get the data
            if start_frame_idx % pred_len == 0:
                data = [get_data(os.path.join(data_vid_filepath, f'{start_frame_idx}.{suf}')), 
                        get_data(os.path.join(data_vid_filepath, f'{start_frame_idx+1}.{suf}'))]
                data = (torch.cat(data, 2)).unsqueeze(0).cuda()
            else:
                data = [get_data(os.path.join(pred_vid_filepath, f'{start_frame_idx}.{suf}')), 
                        get_data(os.path.join(pred_vid_filepath, f'{start_frame_idx+1}.{suf}'))]
                data = (torch.cat(data, 2)).unsqueeze(0).cuda()
            # get the target
            target = [get_data(os.path.join(pred_vid_filepath, f'{start_frame_idx+2}.{suf}')), 
                      get_data(os.path.join(pred_vid_filepath, f'{start_frame_idx+3}.{suf}'))]
            target = (torch.cat(target, 2)).unsqueeze(0).cuda()
            # compute latent space error
            data_state = model.data_to_state(data)
            target_state = model.data_to_state(target)
            output_state = model.model(data_state)
            error_p.append(float(model.loss_func(output_state, target_state).cpu().detach().numpy()))
        errors.append(error_p)
    
    errors = np.array(errors)
    np.save(os.path.join(pred_save_path, 'stability.npy'), errors)


if __name__ == '__main__':
    main()