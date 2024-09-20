from torchvision.utils import save_image
import os
import pytorch_lightning as pl
from utils.misc import mkdir, tuple2name, mkpath
from utils.show import generate_video_directory
from geomloss import SamplesLoss
from typing import Union
from itertools import chain

from models.latent_autoencoder import *
from models.nsv_autoencoder import *
from models.smooth_nsv_autoencoder import *


class VisDynamicsModel(pl.LightningModule):

    def __init__(self,
                 model: None,
                 output_dir: str='outputs',
                 lr: float=1e-4,
                 gamma: float=0.5,
                 lr_schedule: list=[20, 50, 100],
                 reconstruct_loss_type: str='high-dim-latent',
                 smooth_loss_type: str='none',
                 regularize_loss_type: str='none',
                 margin: float=0.0,
                 reconstruct_loss_weight: float=1.0,
                 smooth_loss_weight: Union[list,float]=0.0,
                 regularize_loss_weight: float=0.0,
                 model_annealing_list: list=[],
                 **kwargs) -> None:
        super().__init__()

        self.dt = 1/60 if model.dataset != 'cylindrical_flow' else .02
        self.output_dir = output_dir

        self.model = model
        
        self.loss_func = nn.MSELoss(reduction='none')
        self.regularize_loss_func = SamplesLoss(loss='sinkhorn')

        self.lr = lr
        self.gamma = gamma
        self.lr_schedule = lr_schedule

        self.reconstruct_loss_type = reconstruct_loss_type
        self.smooth_loss_type = smooth_loss_type
        self.regularize_loss_type = regularize_loss_type

        self.reconstruct_loss_weight = reconstruct_loss_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.regularize_loss_weight = regularize_loss_weight 

        self.margin = margin

        self.annealing_list = model_annealing_list
        for s in self.annealing_list:
            self.__dict__[s[0]] = s[2]

        self.save_hyperparameters(ignore=['model'])
    
    
    def forward(self, x):

        if 'smooth' in self.model.name:
            output, latent, state_reconstructured, state, state_gt, latent_gt = self.model(x)
        elif 'base' in self.model.name:
            output, latent, state, latent_gt = self.model(x)
        else:
            output, latent = self.model(x)

        return output
    
    def calc_Losses(self, batch, is_test=False, is_val=False):

        data, output, target, file_tuples, latent_gt, latent, state = None, None, None, None, None, None, None

        if 'smooth' in self.model.name:
            data, target, in_between,  file_tuples = batch

            _, target_latent, target_state_reconstructured, target_state, target_state_gt, target_latent_gt = self.model(target)
            _, in_between_latent, in_between_state_reconstructured, in_between_state, in_between_state_gt, in_between_latent_gt = self.model(in_between)
            output, latent, state_reconstructured, state, state_gt, latent_gt = self.model(data)

            state_clone = state.clone().detach()
            state_max = torch.max(state_clone, dim=0)[0][0]
            state_min = torch.min(state_clone, dim=0)[0][0]

            self.log('state_max{}'.format('_test' if is_test else '_val' if is_val else '_train'), state_max, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('state_min{}'.format('_test' if is_test else '_val' if is_val else '_train'), state_min, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            

            if self.reconstruct_loss_type == 'high-dim-latent':
                
                reconstruct_loss = self.loss_func(latent, latent_gt).sum([1]).mean() 

            self.log('rec{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), reconstruct_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            # smoothness loss
            smooth_loss = torch.as_tensor(0.0, device=self.device)
            if self.smooth_loss_type == 'neighbor-distance':

                data_target_dist = variable_distance(state, target_state)
                smooth_loss = F.relu(data_target_dist - self.margin).mean()
            
            if self.smooth_loss_type == 'neighbor-distance-2':

                data_target_dist = variable_distance(state, target_state)
                data_between_dist = variable_distance(state, in_between_state)

                smooth_loss = F.relu(data_target_dist - self.margin).mean() +  F.relu(data_between_dist - self.margin/2).mean()
            
            self.log('smth{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), smooth_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
            # regularization loss
            regularize_loss = torch.as_tensor(0.0)

            if self.regularize_loss_type == 'sinkhorn':
                # collocation points in [-1, 1]^d
                v_col = 2. * torch.rand(state.shape, device=self.device) - 1.
                regularize_loss = self.regularize_loss_func(state, v_col)

            if self.regularize_loss_type == 'sinkhorn-circle':
                # collocation points in B(0, r)
                radius = 0.8 * torch.rand(state.shape[0], device=self.device)
                theta = 2 * np.pi * torch.rand(state.shape[0], device=self.device)
                v_col = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], 1)
                regularize_loss = self.regularize_loss_func(state, v_col)

            self.log('reg{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), regularize_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
            
            total_loss = self.reconstruct_loss_weight * reconstruct_loss + self.beta * (self.smooth_loss_weight * smooth_loss  \
                                                    + self.regularize_loss_weight * regularize_loss)

        elif 'base' in self.model.name:
            data, target, file_tuples = batch
            output, latent, state, latent_gt = self.model(data)

            state_clone = state.clone().detach()
            state_max = torch.max(state_clone, dim=0)[0][0]
            state_min = torch.min(state_clone, dim=0)[0][0]

            self.log('state_max{}'.format('_test' if is_test else '_val' if is_val else '_train'), state_max, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('state_min{}'.format('_test' if is_test else '_val' if is_val else '_train'), state_min, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
            total_loss = self.loss_func(latent, latent_gt).sum([1]).mean()
            self.log('rec{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        else:
            
            data, target, file_tuples = batch 

            output, latent = self.model(data)
            
            total_loss = self.loss_func(output, target).sum([1,2,3]).mean()
            self.log('rec{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        if is_test:
            self.save_outputs(data, output, target, file_tuples, latent_gt, latent, state)

        return total_loss

    def save_outputs(self, data, output, target, file_tuples, latent_gt, latent, state):
        pxl_loss = self.loss_func(output, target).mean()
        self.log('pxl_rec{}_loss'.format('_test'), pxl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
        self.all_filepaths.extend(file_tuples.cpu().numpy())
        for idx in range(data.shape[0]):
            if 'save_prediction' in self.test_mode:
                comparison = torch.cat([data[idx,:, :, :128].unsqueeze(0),
                                        data[idx,:, :, 128:].unsqueeze(0),
                                        target[idx, :, :, :128].unsqueeze(0),
                                        target[idx, :, :, 128:].unsqueeze(0),
                                        output[idx, :, :, :128].unsqueeze(0),
                                        output[idx, :, :, 128:].unsqueeze(0)])
                mkpath(os.path.join(self.pred_log_dir, str(file_tuples[idx][0].item())))
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir,  tuple2name(file_tuples[idx])), nrow=1)
                self.all_path_nums.add(file_tuples[idx][0].item())

            if 'base' in self.model.name or 'smooth' in self.model.name:
                latent_tmp = latent_gt[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)

                latent_reconstructed_tmp = latent[idx].view(1, -1)[0]
                latent_reconstructed_tmp = latent_reconstructed_tmp.cpu().detach().numpy()
                self.all_reconstructed_latents.append(latent_reconstructed_tmp)

                latent_latent_tmp = state[idx].view(1, -1)[0]
                latent_latent_tmp = latent_latent_tmp.cpu().detach().numpy()
                self.all_refine_latents.append(latent_latent_tmp)
            else:
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)


    def training_step(self, batch, batch_idx):
        
        train_loss = self.calc_Losses(batch)

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('learning rate', self.scheduler.get_lr()[0], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return train_loss

    def validation_step(self, batch, batch_idx):

        val_loss = self.calc_Losses(batch, is_val=True)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return val_loss 
    
    def test_step(self, batch, batch_idx):

        test_loss = self.calc_Losses(batch, is_test=True)

        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return test_loss

    def setup(self, stage=None):

        if stage == 'test':
            # initialize lists for saving variables and latents during testing
            self.all_filepaths = []
            self.all_latents = []
            self.all_refine_latents = []
            self.all_reconstructed_latents = []
            self.all_path_nums = set()

            self.pred_log_dir = os.path.join(self.output_dir, self.model.dataset, self.pred_log_name or "predictions", self.model.name)
            self.var_log_dir = os.path.join(self.output_dir, self.model.dataset, self.var_log_name or "variables", self.model.name)
            if 'save_prediction' in self.test_mode:
                mkdir(self.pred_log_dir)
            mkdir(self.var_log_dir)
    
    def on_test_epoch_end(self) -> None:

        if 'save_prediction' in self.test_mode:
            generate_video_directory(self.pred_log_dir, self.all_path_nums, delete_after=True)

        return super().on_test_epoch_end()
    
    def configure_optimizers(self):

        ae_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(ae_optimizer, milestones=self.lr_schedule, gamma=self.gamma)
        
        return [ae_optimizer], [self.scheduler] 