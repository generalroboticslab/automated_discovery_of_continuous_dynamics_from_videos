from models.sub_modules import *
from models.latent_autoencoder import *
from models.nsv_autoencoder import *
import os
import copy
from collections import OrderedDict
from utils.misc import get_experiment_dim


class SmoothNSVEncoder(torch.nn.Module):
    def __init__(self, extra_layers, nsv_dim=2, **kwargs):
        super(SmoothNSVEncoder, self).__init__()

        self.nsv_dim = nsv_dim
        self.nsv_encoder = NSVEncoder(self.nsv_dim, **kwargs)
        
        self.extra_layers = extra_layers
        
    def forward(self, x):

        nsv, latent_gt = self.nsv_encoder(x)

        nsv_gt = nsv.clone().detach()

        return nsv, nsv_gt, latent_gt

class SmoothNSVDecoder(torch.nn.Module):
    def __init__(self, extra_layers, nsv_dim=2, **kwargs):
        super(SmoothNSVDecoder, self).__init__()
        
        self.nsv_dim = nsv_dim

        self.nsv_decoder = NSVDecoder(self.nsv_dim, **kwargs)

        self.extra_layers = extra_layers
    
    def forward(self, nsv):
        
        output, latent = self.nsv_decoder(nsv)

        return output, latent, nsv


class SmoothNSVAutoencoder(torch.nn.Module):


    @classmethod
    def from_model_name(cls, name, dataset, output_dir, **kwargs):

        params = name.split('_')
        print(params)

        model = cls(dataset, params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]=='True', name, name, output_dir, kwargs['latent_model_name']).eval()

        model.load_smooth_refine_model_weights(name)

        for param in model.parameters():
            param.requires_grad = False

        return model


    def __init__(self, dataset, seed, reconstruct_loss_type, reconstruct_loss_weight, smooth_loss_type, smooth_loss_weight, regularize_loss_type, regularize_loss_weight, annealing, model_name, nsv_model_name, output_dir, latent_model_name="encoder-decoder-64", **kwargs):
        super(SmoothNSVAutoencoder, self).__init__()

        self.name =  '_'.join([model_name, str(seed), reconstruct_loss_type, str(reconstruct_loss_weight), smooth_loss_type, str(smooth_loss_weight), regularize_loss_type, str(regularize_loss_weight), str(annealing)])
        self.dataset = dataset
        self.seed = seed

        self.annealing = annealing
        self.output_dir = output_dir
        self.reconstruct_loss_type = reconstruct_loss_type

        self.nsv_dim = get_experiment_dim(self.dataset, self.seed)

        self.encoder = SmoothNSVEncoder(False, nsv_dim=self.nsv_dim, **kwargs)
        self.decoder = SmoothNSVDecoder(False, nsv_dim=self.nsv_dim, **kwargs)
        
        self.latent_model_name = latent_model_name

        self.load_hyper_model_weights()

    def load_hyper_model_weights(self,):

        weight_dir = os.getcwd() + '/' + self.output_dir+ '/'  + self.dataset + "/checkpoints/" + '_'.join([self.latent_model_name, str(self.seed)])
        items = os.listdir(weight_dir)
        for i in items:
            if i != "last.ckpt":
                weight_path = os.path.join(weight_dir, i)

        ckpt = torch.load(weight_path)

        hyper_model = LatentAutoEncoder(3, self.dataset, self.seed)

        renamed_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            name = k.replace('model.', '')
            if name in ['mu', 'std']:
                print('PREVIOUS VERSION!!!')
                pass
            else:
                renamed_state_dict[name] = v

        hyper_model.load_state_dict(renamed_state_dict)

        self.encoder.nsv_encoder.latent_encoder = copy.deepcopy(hyper_model.encoder)
        self.decoder.nsv_decoder.latent_decoder = copy.deepcopy(hyper_model.decoder)

        if self.reconstruct_loss_type != 'output':
            for param in self.encoder.nsv_encoder.latent_encoder.parameters():
                param.requires_grad = False
            self.encoder.nsv_encoder.latent_encoder.eval()
            for param in self.decoder.nsv_decoder.latent_decoder.parameters():
                param.requires_grad = False
            self.decoder.nsv_decoder.latent_decoder.eval()

    def load_smooth_refine_model_weights(self, smooth_model_name):

        weight_dir = os.getcwd() + '/' + self.output_dir + '/' + self.dataset + "/checkpoints/" + smooth_model_name
        items = os.listdir(weight_dir)
        for i in items:
            if i != "last.ckpt":
                weight_path = os.path.join(weight_dir, i)

        print("Weight path smooth model: ", weight_path)

        ckpt = torch.load(weight_path)

        renamed_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            name = k.replace('model.', '')

            renamed_state_dict[name] = v

        self.load_state_dict(renamed_state_dict)
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.encoder.eval()
        self.decoder.eval()

    def forward(self, x):

        state, state_gt, latent_gt = self.encoder(x)

        state = torch.reshape(state, (-1,state.size(-1)))

        output, latent, state_reconstructured = self.decoder(state)

        return output, latent, state_reconstructured, state, state_gt, latent_gt