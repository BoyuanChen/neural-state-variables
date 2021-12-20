
import os
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
from dataset import NeuralPhysDataset
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
                         RefineReactionDiffusionModel)


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

class VisDynamicsModel(pl.LightningModule):

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
        if self.hparams.model_name == 'encoder-decoder':
            self.model = EncoderDecoder(in_channels=3)
        if self.hparams.model_name == 'encoder-decoder-64':
            self.model = EncoderDecoder64x1x1(in_channels=3)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'single_pendulum':
            self.model = RefineSinglePendulumModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'double_pendulum':
            self.model = RefineDoublePendulumModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'circular_motion':
            self.model = RefineCircularMotionModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'swingstick_non_magnetic':
            self.model = RefineSwingStickNonMagneticModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'air_dancer':
            self.model = RefineAirDancerModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'lava_lamp':
            self.model = RefineLavaLampModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'fire':
            self.model = RefineFireModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'elastic_pendulum':
            self.model = RefineElasticPendulumModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'reaction_diffusion':
            self.model = RefineReactionDiffusionModel(in_channels=64)
        if 'refine' in self.hparams.model_name and self.hparams.if_test:
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
        # loss
        self.loss_func = nn.MSELoss()

    def train_forward(self, x):
        if self.hparams.model_name == 'encoder-decoder' or 'refine' in self.hparams.model_name:
            output, latent = self.model(x)
        if self.hparams.model_name == 'encoder-decoder-64':
            output, latent = self.model(x, x, False)
        return output, latent

    def training_step(self, batch, batch_idx):
        data, target, filepath = batch
        output, latent = self.train_forward(data)

        train_loss = self.loss_func(output, target)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target, filepath = batch
        output, latent = self.train_forward(data)

        val_loss = self.loss_func(output, target)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        
        if self.hparams.model_name == 'encoder-decoder' or self.hparams.model_name == 'encoder-decoder-64':
            data, target, filepath = batch
            if self.hparams.model_name == 'encoder-decoder':
                output, latent = self.model(data)
            if self.hparams.model_name == 'encoder-decoder-64':
                output, latent = self.model(data, data, False)
            test_loss = self.loss_func(output, target)
            self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # save the output images and latent vectors
            self.all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                comparison = torch.cat([data[idx,:, :, :128].unsqueeze(0),
                                        data[idx,:, :, 128:].unsqueeze(0),
                                        target[idx, :, :, :128].unsqueeze(0),
                                        target[idx, :, :, 128:].unsqueeze(0),
                                        output[idx, :, :, :128].unsqueeze(0),
                                        output[idx, :, :, 128:].unsqueeze(0)])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)

        if 'refine' in self.hparams.model_name:
            data, target, filepath = batch
            _, latent = self.high_dim_model(data, data, False)
            latent = latent.squeeze(-1).squeeze(-1)
            latent_reconstructed, latent_latent = self.model(latent)
            output, _ = self.high_dim_model(data, latent_reconstructed.unsqueeze(2).unsqueeze(3), True)
            # calculate losses
            pixel_reconstruction_loss = self.loss_func(output, target)
            test_loss = self.loss_func(latent_reconstructed, latent)
            self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('pixel_reconstruction_loss', pixel_reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # save the output images and latent vectors
            self.all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                comparison = torch.cat([data[idx, :, :, :128].unsqueeze(0),
                                        data[idx, :, :, 128:].unsqueeze(0),
                                        target[idx, :, :, :128].unsqueeze(0),
                                        target[idx, :, :, 128:].unsqueeze(0),
                                        output[idx, :, :, :128].unsqueeze(0),
                                        output[idx, :, :, 128:].unsqueeze(0)])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)
                # save latent_latent: the latent vector in the refine network
                latent_latent_tmp = latent_latent[idx].view(1, -1)[0]
                latent_latent_tmp = latent_latent_tmp.cpu().detach().numpy()
                self.all_refine_latents.append(latent_latent_tmp)
                # save latent_reconstructed: the latent vector reconstructed by the entire refine network
                latent_reconstructed_tmp = latent_reconstructed[idx].view(1, -1)[0]
                latent_reconstructed_tmp = latent_reconstructed_tmp.cpu().detach().numpy()
                self.all_reconstructed_latents.append(latent_reconstructed_tmp)


    def test_save(self):
        if self.hparams.model_name == 'encoder-decoder' or self.hparams.model_name == 'encoder-decoder-64':
            np.save(os.path.join(self.var_log_dir, 'ids.npy'), self.all_filepaths)
            np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)
        if 'refine' in self.hparams.model_name:
            np.save(os.path.join(self.var_log_dir, 'ids.npy'), self.all_filepaths)
            np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)
            np.save(os.path.join(self.var_log_dir, 'refine_latent.npy'), self.all_refine_latents)
            np.save(os.path.join(self.var_log_dir, 'reconstructed_latent.npy'), self.all_reconstructed_latents)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
    
    def paths_to_tuple(self, paths):
        new_paths = []
        for i in range(len(paths)):
            tmp = paths[i].split('.')[0].split('_')
            new_paths.append((int(tmp[0]), int(tmp[1])))
        return new_paths

    def setup(self, stage=None):

        if stage == 'fit':
            # for the training of the refine network, we need to have the latent data as the dataset
            if 'refine' in self.hparams.model_name:
                high_dim_var_log_dir = self.var_log_dir.replace('refine', 'encoder-decoder')
                train_data = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_train', 'latent.npy')))
                train_target = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_train', 'latent.npy')))
                val_data = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'latent.npy')))
                val_target = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'latent.npy')))
                train_filepaths = list(np.load(os.path.join(high_dim_var_log_dir+'_train', 'ids.npy')))
                val_filepaths = list(np.load(os.path.join(high_dim_var_log_dir+'_val', 'ids.npy')))
                # convert the file strings into tuple so that we can use TensorDataset to load everything together
                train_filepaths = torch.Tensor(self.paths_to_tuple(train_filepaths))
                val_filepaths = torch.Tensor(self.paths_to_tuple(val_filepaths))
                self.train_dataset = torch.utils.data.TensorDataset(train_data, train_target, train_filepaths)
                self.val_dataset = torch.utils.data.TensorDataset(val_data, val_target, val_filepaths)
            else:
                self.train_dataset = NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                       flag='train',
                                                       seed=self.hparams.seed,
                                                       object_name=self.hparams.dataset)
                self.val_dataset = NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                     flag='val',
                                                     seed=self.hparams.seed,
                                                     object_name=self.hparams.dataset)

        if stage == 'test':
            self.test_dataset = NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                  flag='test',
                                                  seed=self.hparams.seed,
                                                  object_name=self.hparams.dataset)
            
            # initialize lists for saving variables and latents during testing
            self.all_filepaths = []
            self.all_latents = []
            self.all_refine_latents = []
            self.all_reconstructed_latents = []

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