

import os
import sys
import glob
import yaml
import torch
import pprint
import shutil
import numpy as np
from tqdm import tqdm
from munch import munchify
from collections import OrderedDict
from models import VisDynamicsModel
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


def main():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = VisDynamicsModel(lr=cfg.lr,
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

    ckpt = torch.load(checkpoint_filepath)
    if 'refine' in cfg.model_name:
        ckpt = rename_ckpt_for_multi_models(ckpt)
        model.model.load_state_dict(ckpt)

        high_dim_checkpoint_filepath = str(sys.argv[3])
        high_dim_checkpoint_filepath = glob.glob(os.path.join(high_dim_checkpoint_filepath, '*.ckpt'))[0]
        ckpt = torch.load(high_dim_checkpoint_filepath)
        ckpt = rename_ckpt_for_multi_models(ckpt)
        model.high_dim_model.load_state_dict(ckpt)

    else:
        model.load_state_dict(ckpt['state_dict'])

    model.eval()
    model.freeze()

    trainer = Trainer(gpus=1,
                      deterministic=True,
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0)

    trainer.test(model)
    model.test_save()


def main_latentpred():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    high_dim_checkpoint_filepath = str(sys.argv[3])
    refine_checkpoint_filepath = str(sys.argv[4])
    
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
    model.eval()
    model.freeze()

    trainer = Trainer(gpus=1,
                      deterministic=True,
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0)

    trainer.test(model)


def rename_ckpt_for_multi_models(ckpt):
    renamed_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if 'high_dim_model' in k:
            name = k.replace('high_dim_model.', '')
        else:
            name = k.replace('model.', '')
        renamed_state_dict[name] = v
    return renamed_state_dict

# gather latent variables by running training data on the trained high-dim models
def gather_latent_from_trained_high_dim_model():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = VisDynamicsModel(lr=cfg.lr,
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

    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')
    model.eval()
    model.freeze()

    # prepare train and val dataset
    kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True} if cfg.if_cuda else {}
    train_dataset = NeuralPhysDataset(data_filepath=cfg.data_filepath,
                                      flag='train',
                                      seed=cfg.seed,
                                      object_name=cfg.dataset)
    val_dataset = NeuralPhysDataset(data_filepath=cfg.data_filepath,
                                    flag='val',
                                    seed=cfg.seed,
                                    object_name=cfg.dataset)
    # prepare train and val loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=cfg.train_batch,
                                               shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=cfg.val_batch,
                                             shuffle=False,
                                             **kwargs)

    # run train forward pass to save the latent vector for training the refine network later
    all_filepaths = []
    all_latents = []
    var_log_dir = os.path.join(log_dir, 'variables')
    for batch_idx, (data, target, filepath) in enumerate(tqdm(train_loader)):
        if cfg.model_name == 'encoder-decoder':
            output, latent = model.model(data.cuda())
        if cfg.model_name == 'encoder-decoder-64':
            output, latent = model.model(data.cuda(), data.cuda(), False)
        # save the latent vectors
        all_filepaths.extend(filepath)
        for idx in range(data.shape[0]):
            latent_tmp = latent[idx].view(1, -1)[0]
            latent_tmp = latent_tmp.cpu().detach().numpy()
            all_latents.append(latent_tmp)

    mkdir(var_log_dir+'_train')
    np.save(os.path.join(var_log_dir+'_train', 'ids.npy'), all_filepaths)
    np.save(os.path.join(var_log_dir+'_train', 'latent.npy'), all_latents)

    # run val forward pass to save the latent vector for validating the refine network later
    all_filepaths = []
    all_latents = []
    var_log_dir = os.path.join(log_dir, 'variables')
    for batch_idx, (data, target, filepath) in enumerate(tqdm(val_loader)):
        if cfg.model_name == 'encoder-decoder':
            output, latent = model.model(data.cuda())
        if cfg.model_name == 'encoder-decoder-64':
            output, latent = model.model(data.cuda(), data.cuda(), False)
        # save the latent vectors
        all_filepaths.extend(filepath)
        for idx in range(data.shape[0]):
            latent_tmp = latent[idx].view(1, -1)[0]
            latent_tmp = latent_tmp.cpu().detach().numpy()
            all_latents.append(latent_tmp)

    mkdir(var_log_dir+'_val')
    np.save(os.path.join(var_log_dir+'_val', 'ids.npy'), all_filepaths)
    np.save(os.path.join(var_log_dir+'_val', 'latent.npy'), all_latents)


# gather latent variables by running training data on the trained refine models
def gather_latent_from_trained_refine_model():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = VisDynamicsModel(lr=cfg.lr,
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

    ckpt = torch.load(checkpoint_filepath)
    ckpt = rename_ckpt_for_multi_models(ckpt)
    model.model.load_state_dict(ckpt)
    high_dim_checkpoint_filepath = str(sys.argv[3])
    high_dim_checkpoint_filepath = glob.glob(os.path.join(high_dim_checkpoint_filepath, '*.ckpt'))[0]
    ckpt = torch.load(high_dim_checkpoint_filepath)
    ckpt = rename_ckpt_for_multi_models(ckpt)
    model.high_dim_model.load_state_dict(ckpt)
    model = model.to('cuda')
    model.eval()
    model.freeze()

    # prepare train and val dataset
    kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True} if cfg.if_cuda else {}
    train_dataset = NeuralPhysDataset(data_filepath=cfg.data_filepath,
                                      flag='train',
                                      seed=cfg.seed,
                                      object_name=cfg.dataset)
    val_dataset = NeuralPhysDataset(data_filepath=cfg.data_filepath,
                                    flag='val',
                                    seed=cfg.seed,
                                    object_name=cfg.dataset)
    # prepare train and val loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=cfg.train_batch,
                                               shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=cfg.val_batch,
                                             shuffle=False,
                                             **kwargs)

    # run train forward pass to save the latent vector for training data
    all_filepaths = []
    all_latents = []
    all_refine_latents = []
    all_reconstructed_latents = []
    var_log_dir = os.path.join(log_dir, 'variables')
    for batch_idx, (data, target, filepath) in enumerate(tqdm(train_loader)):
        _, latent = model.high_dim_model(data.cuda(), data.cuda(), False)
        latent = latent.squeeze(-1).squeeze(-1)
        latent_reconstructed, latent_latent = model.model(latent)
        # save the latent vectors
        all_filepaths.extend(filepath)
        for idx in range(data.shape[0]):
            latent_tmp = latent[idx].view(1, -1)[0]
            latent_tmp = latent_tmp.cpu().detach().numpy()
            all_latents.append(latent_tmp)
            # save latent_latent: the latent vector in the refine network
            latent_latent_tmp = latent_latent[idx].view(1, -1)[0]
            latent_latent_tmp = latent_latent_tmp.cpu().detach().numpy()
            all_refine_latents.append(latent_latent_tmp)
            # save latent_reconstructed: the latent vector reconstructed by the entire refine network
            latent_reconstructed_tmp = latent_reconstructed[idx].view(1, -1)[0]
            latent_reconstructed_tmp = latent_reconstructed_tmp.cpu().detach().numpy()
            all_reconstructed_latents.append(latent_reconstructed_tmp)

    mkdir(var_log_dir+'_train')
    np.save(os.path.join(var_log_dir+'_train', 'ids.npy'), all_filepaths)
    np.save(os.path.join(var_log_dir+'_train', 'latent.npy'), all_latents)
    np.save(os.path.join(var_log_dir+'_train', 'refine_latent.npy'), all_refine_latents)
    np.save(os.path.join(var_log_dir+'_train', 'reconstructed_latent.npy'), all_reconstructed_latents)

    # run val forward pass to save the latent vector for validation data
    all_filepaths = []
    all_latents = []
    all_refine_latents = []
    all_reconstructed_latents = []
    var_log_dir = os.path.join(log_dir, 'variables')
    for batch_idx, (data, target, filepath) in enumerate(tqdm(val_loader)):
        _, latent = model.high_dim_model(data.cuda(), data.cuda(), False)
        latent = latent.squeeze(-1).squeeze(-1)
        latent_reconstructed, latent_latent = model.model(latent)
        # save the latent vectors
        all_filepaths.extend(filepath)
        for idx in range(data.shape[0]):
            latent_tmp = latent[idx].view(1, -1)[0]
            latent_tmp = latent_tmp.cpu().detach().numpy()
            all_latents.append(latent_tmp)
            # save latent_latent: the latent vector in the refine network
            latent_latent_tmp = latent_latent[idx].view(1, -1)[0]
            latent_latent_tmp = latent_latent_tmp.cpu().detach().numpy()
            all_refine_latents.append(latent_latent_tmp)
            # save latent_reconstructed: the latent vector reconstructed by the entire refine network
            latent_reconstructed_tmp = latent_reconstructed[idx].view(1, -1)[0]
            latent_reconstructed_tmp = latent_reconstructed_tmp.cpu().detach().numpy()
            all_reconstructed_latents.append(latent_reconstructed_tmp)

    mkdir(var_log_dir+'_val')
    np.save(os.path.join(var_log_dir+'_val', 'ids.npy'), all_filepaths)
    np.save(os.path.join(var_log_dir+'_val', 'latent.npy'), all_latents)
    np.save(os.path.join(var_log_dir+'_val', 'refine_latent.npy'), all_refine_latents)
    np.save(os.path.join(var_log_dir+'_val', 'reconstructed_latent.npy'), all_reconstructed_latents)


if __name__ == '__main__':
    if str(sys.argv[4]) == 'eval-train':
        gather_latent_from_trained_high_dim_model()
    elif str(sys.argv[4]) == 'eval-refine-train':
        gather_latent_from_trained_refine_model()
    elif str(sys.argv[5]) == 'eval-latentpred':
        main_latentpred()
    else:
        main()