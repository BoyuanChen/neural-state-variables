
import os
import sys
import yaml
import torch
import pprint
from munch import munchify
from models import VisDynamicsModel
from models_latentpred import VisLatentDynamicsModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


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
                             if_test=False,
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

    # define callback for selecting checkpoints during training
    checkpoint_callback = ModelCheckpoint(
        filepath=log_dir + "/lightning_logs/checkpoints/{epoch}_{val_loss}",
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='')

    # define trainer
    trainer = Trainer(gpus=cfg.num_gpus,
                      max_epochs=cfg.epochs,
                      deterministic=True,
                      accelerator='ddp',
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0,
                      checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

def main_latentpred():
    config_filepath = str(sys.argv[2])
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
                                   if_test=False,
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

    model.load_high_dim_refine_model(high_dim_checkpoint_filepath, refine_checkpoint_filepath)

    # define callback for selecting checkpoints during training
    checkpoint_callback = ModelCheckpoint(
        filepath=log_dir + "/lightning_logs/checkpoints/{epoch}_{val_loss}",
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='')

    # define trainer
    trainer = Trainer(gpus=cfg.num_gpus,
                      max_epochs=cfg.epochs,
                      deterministic=True,
                      accelerator='ddp',
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0,
                      checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

if __name__ == '__main__':
    if sys.argv[1] == 'latentpred':
        main_latentpred()
    else:
        main()
