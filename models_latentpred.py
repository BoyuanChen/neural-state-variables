
import os
import glob
import torch
import shutil
import numpy as np
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import NeuralPhysDataset, NeuralPhysLatentDynamicsDataset
from model_utils import (EncoderDecoder,
                         EncoderDecoder64x1x1,
                         RefineDoublePendulumModel,
                         RefineSinglePendulumModel,
                         RefineCircularMotionModel,
                         RefineModelReLU,
                         RefineSwingStickNonMagneticModel,
                         RefineAirDancerModel,
                         RefineLavaLampModel,
                         RefineFireModel,
                         RefineElasticPendulumModel,
                         RefineReactionDiffusionModel,
                         LatentPredModel)


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def rename_ckpt_for_multi_models(ckpt):
    renamed_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if k.split('.')[0] == 'model':
            name = k.replace('model.', '')
            renamed_state_dict[name] = v
    return renamed_state_dict

class VisLatentDynamicsModel(pl.LightningModule):

    def __init__(self,
                 lr: float=1e-4,
                 seed: int=1,
                 if_cuda: bool=True,
                 if_test: bool=False,
                 gamma: float=0.5,
                 log_dir: str='logs',
                 train_batch: int=512,
                 val_batch: int=256,
                 test_batch: int=256,
                 num_workers: int=8,
                 model_name: str='encoder-decoder-64',
                 data_filepath: str='data',
                 dataset: str='single_pendulum',
                 lr_schedule: list=[20, 50, 100]) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}
        # create visualization saving folder if testing
        self.pred_log_dir = os.path.join(self.hparams.log_dir, 'predictions')
        self.var_log_dir = os.path.join(self.hparams.log_dir, 'variables')
        if not self.hparams.if_test:
            mkdir(self.pred_log_dir)
            mkdir(self.var_log_dir)

        self.__build_model()

    def __build_model(self):
        # model
        if self.hparams.model_name == 'latent-prediction' and self.hparams.dataset == 'single_pendulum':
            self.model = LatentPredModel(in_channels=2)
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
            self.refine_model = RefineSinglePendulumModel(in_channels=64)
        if self.hparams.model_name == 'latent-prediction' and self.hparams.dataset == 'double_pendulum':
            self.model = LatentPredModel(in_channels=4)
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
            self.refine_model = RefineDoublePendulumModel(in_channels=64)
        if self.hparams.model_name == 'latent-prediction' and self.hparams.dataset == 'elastic_pendulum':
            self.model = LatentPredModel(in_channels=6)
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
            self.refine_model = RefineElasticPendulumModel(in_channels=64)
        if self.hparams.model_name == 'latent-prediction' and self.hparams.dataset == 'swingstick_non_magnetic':
            self.model = LatentPredModel(in_channels=4)
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
            self.refine_model = RefineSwingStickNonMagneticModel(in_channels=64)
        if self.hparams.model_name == 'latent-prediction' and self.hparams.dataset == 'air_dancer':
            self.model = LatentPredModel(in_channels=8)
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
            self.refine_model = RefineAirDancerModel(in_channels=64)
        if self.hparams.model_name == 'latent-prediction' and self.hparams.dataset == 'lava_lamp':
            self.model = LatentPredModel(in_channels=8)
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
            self.refine_model = RefineLavaLampModel(in_channels=64)
        if self.hparams.model_name == 'latent-prediction' and self.hparams.dataset == 'fire':
            self.model = LatentPredModel(in_channels=24)
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
            self.refine_model = RefineFireModel(in_channels=64)
        if self.hparams.model_name == 'latent-prediction' and self.hparams.dataset == 'reaction_diffusion':
            self.model = LatentPredModel(in_channels=2)
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
            self.refine_model = RefineReactionDiffusionModel(in_channels=64)
        # loss
        self.loss_func = nn.MSELoss()
    
    def load_model(self, checkpoint_filepath):
        # load model for test
        checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
        ckpt = torch.load(checkpoint_filepath)
        ckpt = rename_ckpt_for_multi_models(ckpt)
        self.model.load_state_dict(ckpt)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
    
    def load_high_dim_refine_model(self, high_dim_checkpoint_filepath, refine_checkpoint_filepath):
        # load high-dim and refine models
        high_dim_checkpoint_filepath = glob.glob(os.path.join(high_dim_checkpoint_filepath, '*.ckpt'))[0]
        ckpt = torch.load(high_dim_checkpoint_filepath)
        ckpt = rename_ckpt_for_multi_models(ckpt)
        self.high_dim_model.load_state_dict(ckpt)

        refine_checkpoint_filepath = glob.glob(os.path.join(refine_checkpoint_filepath, '*.ckpt'))[0]
        ckpt = torch.load(refine_checkpoint_filepath)
        ckpt = rename_ckpt_for_multi_models(ckpt)
        self.refine_model.load_state_dict(ckpt)

        for p in self.high_dim_model.parameters():
            p.requires_grad = False
        self.high_dim_model.eval()

        for p in self.refine_model.parameters():
            p.requires_grad = False
        self.refine_model.eval()
    
    def extract_decoder_from_refine_model(self):
        _layers = list(self.refine_model.children())[4:]
        self.refine_model_decoder = torch.nn.Sequential(*_layers)

        for p in self.refine_model_decoder.parameters():
            p.requires_grad = False
        self.refine_model_decoder.eval()

    def data_to_state(self, data):
        # (B, 3, 128, 256) -> (B, 64, 1, 1) -> (B, 64) -> (B, ID)
        _, latent = self.high_dim_model(data, data, False)
        latent = latent.squeeze(-1).squeeze(-1)
        _, state = self.refine_model(latent)
        return state

    def training_step(self, batch, batch_idx):
        data, target, filepath = batch
        data_state = self.data_to_state(data)
        target_state = self.data_to_state(target)
        output_state = self.model(data_state)

        train_loss = self.loss_func(output_state, target_state)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target, filepath = batch
        data_state = self.data_to_state(data)
        target_state = self.data_to_state(target)
        output_state = self.model(data_state)

        val_loss = self.loss_func(output_state, target_state)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        data, target, target_target, filepath = batch
        data_state = self.data_to_state(data)
        target_state = self.data_to_state(target)
        output_state = self.model(data_state)
        
        latent_reconstructed = self.refine_model_decoder(output_state)
        output, _ = self.high_dim_model(data, latent_reconstructed.unsqueeze(2).unsqueeze(3), True)

        # calculate losses
        test_loss = self.loss_func(output_state, target_state)
        pixel_reconstruction_loss = self.loss_func(output, target_target)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('pixel_reconstruction_loss', pixel_reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # save the output images
        for idx in range(data.shape[0]):
            comparison = torch.cat([data[idx, :, :, :128].unsqueeze(0),
                                    data[idx, :, :, 128:].unsqueeze(0),
                                    target[idx, :, :, :128].unsqueeze(0),
                                    target[idx, :, :, 128:].unsqueeze(0),
                                    target_target[idx, :, :, :128].unsqueeze(0),
                                    target_target[idx, :, :, 128:].unsqueeze(0),
                                    output[idx, :, :, :128].unsqueeze(0),
                                    output[idx, :, :, 128:].unsqueeze(0)])
            save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                    flag='train',
                                                    seed=self.hparams.seed,
                                                    object_name=self.hparams.dataset)
            self.val_dataset = NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                    flag='val',
                                                    seed=self.hparams.seed,
                                                    object_name=self.hparams.dataset)

        if stage == 'test':
            self.test_dataset = NeuralPhysLatentDynamicsDataset(data_filepath=self.hparams.data_filepath,
                                                                 flag='test',
                                                                 seed=self.hparams.seed,
                                                                 object_name=self.hparams.dataset)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.hparams.train_batch,
                                                   shuffle=True,
                                                   **self.kwargs)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                 batch_size=self.hparams.val_batch,
                                                 shuffle=False,
                                                 **self.kwargs)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.hparams.test_batch,
                                                  shuffle=False,
                                                  **self.kwargs)
        return test_loader