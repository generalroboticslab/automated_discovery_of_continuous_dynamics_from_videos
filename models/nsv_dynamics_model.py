import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchdyn.core import NeuralODE
from models.smooth_nsv_autoencoder import *

from itertools import chain
from geomloss import SamplesLoss
from PIL import Image
import IPython
from utils.misc import create_name, tuple2name, mkdir


class NSVDynamicsModel(pl.LightningModule):
    def __init__(self,
                 model: None,
                 nsv_model: None,
                 model_name: str="regress",
                 reconstruct_loss_weight: float=1.0,
                 regularize_loss_weight: float=.0001,
                 pred_length: int=56,
                 dataset: str="single_pendulum",
                 lr: float=1e-4,
                 gamma: float=0.5,
                 lr_schedule: list=[20, 50, 100],
                 output_dir: str="outputs",
                 seed: int=1,
                 extra_steps: int=0,
                 model_annealing_list: list=[],
                 dt: float=1/60,
                 **kwargs) -> None:
        super().__init__()

        self.loss_func = nn.MSELoss(reduction='none')

        self.model = model
        self.nsv_model= nsv_model
        self.model_name = model_name

        self.reconstruct_loss_weight = reconstruct_loss_weight
        self.pred_length = pred_length
        self.extra_steps = extra_steps

        self.ode = NeuralODE(self.model, solver='rk4')
            
        self.dataset = dataset
        self.lr = lr
        self.gamma = gamma
        self.lr_schedule = lr_schedule
        self.dt = dt
        self.output_dir = output_dir
        
        self.annealing_list = model_annealing_list
        for s in self.annealing_list:
            self.__dict__[s[0]] = s[2]
        self.save_hyperparameters(ignore=['model', 'nsv_model'])
    

    def calc_loss(self, batch, is_test=False, is_val=False):
        data, target, weight, file_tuple = batch

        if 'discrete' in self.model_name:
    
            target_fd = ((target[:, 0] - data[:, self.extra_steps:]) / self.dt)

            output = self.model(data.clone())

            reconstruct_loss = ((output - target_fd)**2).sum(1).mean()

        else:
            t_span = (self.dt * torch.arange(self.pred_length+1)).float()
            _, pred = self.ode(data.clone(), t_span)
            pred = pred[1:].permute(1, 0, 2)

            weight = weight.expand(-1,-1,pred.shape[-1])
            reconstruct_loss = (((pred - target)**2) * weight).sum([1,2]).mean()
        
        self.log('rec{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), reconstruct_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)  
        
        total_loss = self.reconstruct_loss_weight * reconstruct_loss
       
        if is_test:

            if 'discrete' in self.model_name:
                t_span = (self.dt * torch.arange(self.pred_length+1)).float()
                _, pred = self.ode(data.clone(), t_span)
                pred = pred[1:].permute(1, 0, 2)

            if 'smooth' in self.nsv_model.name:
                self.nsv_model.eval()
                pred_output, _ = self.nsv_model.decoder.nsv_decoder(pred[:,0,:])
            else:
                self.nsv_model.eval()
                pred_output, _ = self.nsv_model.decoder(pred[:,0,:])
            
            target_output = self.get_target_image(file_tuple)

            pxl_rec_loss = self.loss_func(pred_output, target_output).mean()
            self.log('pxl_rec_test_loss', pxl_rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    

            output = self.model(data.clone())
            self.all_data.extend(data.cpu().detach().numpy())
            self.all_outputs.extend(output.cpu().detach().numpy())

            self.all_preds.extend(pred.cpu().detach().numpy())
            self.all_targets.extend(target.cpu().detach().numpy())
            self.all_filepaths.extend(file_tuple.cpu().detach().numpy())
           

        return total_loss

    def get_target_image(self, file_tuple):
        

        def get_data(filepath):
            data = Image.open(filepath)
            data = data.resize((128, 128))
            data = np.array(data)
            data = torch.tensor(data / 255.0)
            data = data.permute(2, 0, 1).float()
            return data.cuda()
         
        target_output = []
        file_path = os.path.join('data', self.dataset)

        for idx in range(file_tuple.shape[0]):

           suf = os.listdir(os.path.join(file_path,f'{file_tuple[idx][0]}'))[0].split('.')[-1]
           filename1 = f'{file_tuple[idx][0]}/{file_tuple[idx][1]+3}.{suf}'
           filename2 = f'{file_tuple[idx][0]}/{file_tuple[idx][1]+4}.{suf}' 
           target_output.append(torch.cat([get_data(os.path.join(file_path, filename1)), get_data(os.path.join(file_path, filename2))], dim=-1))
        
        target_output = torch.stack(target_output)

        return target_output


    def training_step(self, batch, batch_idx):
        
        train_loss = self.calc_loss(batch)
            
        self.log('learning rate', self.scheduler.get_lr()[0], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
        return train_loss
    

    def validation_step(self, batch, batch_idx):

        val_loss = self.calc_loss(batch, is_val=True)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        
        test_loss = self.calc_loss(batch, is_test=True)

        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return test_loss

    def setup(self, stage=None):

        if self.nsv_model:
            self.nsv_model.eval()

        if stage == 'test':
            self.all_filepaths = []
            self.all_data = []
            self.all_outputs = []
            self.all_preds = []
            self.all_targets = []

            self.var_log_dir = os.path.join(self.output_dir, self.dataset, self.var_log_name or "variables", self.model_name)
            mkdir(self.var_log_dir)
            


    def configure_optimizers(self):

        ode_optimizer = torch.optim.Adam(self.ode.parameters(), lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(ode_optimizer, milestones=self.lr_schedule, gamma=self.gamma)
        
        return [ode_optimizer], [self.scheduler] 