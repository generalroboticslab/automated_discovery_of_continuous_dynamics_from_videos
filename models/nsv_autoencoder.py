from models.sub_modules import *
from models.latent_autoencoder import *
import os
import copy
from collections import OrderedDict
from utils.misc import get_experiment_dim


class NSVEncoder(torch.nn.Module):
    def __init__(self, nsv_dim=2, architecture:str="default", method:str="default", **kwargs):
        super(NSVEncoder, self).__init__()

        self.latent_encoder = LatentEncoder(in_channels=3)
        self.nsv_dim = nsv_dim
        self.architecture = architecture
        self.method = method

        self.layer1 = SirenLayer(64, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, self.nsv_dim)
    
    def latent_forward(self, latent_gt):

        state = self.layer1(latent_gt)
        state = self.layer2(state)
        state = self.layer3(state)
        state = self.layer4(state)

        return state


    def forward(self, x):

        latent_gt = self.latent_encoder(x)
        latent_gt = torch.squeeze(latent_gt)

        if len(latent_gt.shape) < 2:
            latent_gt = torch.unsqueeze(latent_gt, 0)
        
        state = self.layer1(latent_gt)
        state = self.layer2(state)
        state = self.layer3(state)
        state = self.layer4(state)

        return state, latent_gt.detach()

class NSVDecoder(torch.nn.Module):
    def __init__(self, nsv_dim=2, architecture:str="default", method:str="default", **kwargs):
        super(NSVDecoder, self).__init__()

        self.latent_decoder = LatentDecoder(out_channels=3)
        self.nsv_dim = nsv_dim
        self.architecture = architecture
        self.method = method

        self.layer5 = SirenLayer(self.nsv_dim, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, 64, is_last=True)
    
    def forward(self, state):

        latent = self.layer5(state)
        latent = self.layer6(latent)
        latent = self.layer7(latent)
        latent = self.layer8(latent)

        output = torch.unsqueeze(latent, dim=-1)
        output = torch.unsqueeze(output, dim=-1)

        output = self.latent_decoder(output)

        return output, latent


class NSVAutoencoder(torch.nn.Module):

    @classmethod
    def from_model_name(cls, name, dataset, output_dir, **kwargs):

        params = name.split('_')

        seed = params[-1]

        model = cls(dataset, seed, params[0], output_dir)
        model.load_refine_model_weights(name)

        for param in model.parameters():
            param.requires_grad = False

        return model

    def __init__(self, dataset, seed, model_name, output_dir, freeze_hyper=True, architecture="default", method="default", **kwargs):
        super(NSVAutoencoder, self).__init__()

        self.name = '_'.join([model_name, str(seed)])
        self.dataset = dataset
        self.seed = seed
        self.output_dir = output_dir

        self.architecture = architecture
        self.method = method

        self.nsv_dim = get_experiment_dim(self.dataset, self.seed)

        self.encoder = NSVEncoder(self.nsv_dim, architecture, method, **kwargs)
        self.decoder = NSVDecoder(self.nsv_dim, architecture, **kwargs)

        self.freeze_hyper = freeze_hyper

        self.load_hyper_model_weights()

    def load_refine_model_weights(self, name):

        renamed_state_dict = OrderedDict()

        weight_dir = os.getcwd() + '/' + self.output_dir + '/' + self.dataset + "/checkpoints/" + name
        items = os.listdir(weight_dir)
        for i in items:
            if i != "last.ckpt":
                weight_path = os.path.join(weight_dir, i)
        ckpt = torch.load(weight_path)

        for k, v in ckpt['state_dict'].items():

            k = k.replace('model.','')
            
            renamed_state_dict[k] = v

        self.load_state_dict(renamed_state_dict)
    
    def load_hyper_model_weights(self,):

        weight_dir = os.getcwd() + '/' + self.output_dir+ '/'  + self.dataset + "/checkpoints/" + '_'.join(["encoder-decoder-64", str(self.seed)])
        items = os.listdir(weight_dir)
        for i in items:
            if i != "last.ckpt":
                weight_path = os.path.join(weight_dir, i)

        ckpt = torch.load(weight_path)

        hyper_model = LatentAutoEncoder(3, self.dataset, self.seed)

        renamed_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            name = k.replace('model.', '')
            renamed_state_dict[name] = v

        hyper_model.load_state_dict(renamed_state_dict)

        self.encoder.latent_encoder = copy.deepcopy(hyper_model.encoder)
        self.decoder.latent_decoder = copy.deepcopy(hyper_model.decoder)

        if self.freeze_hyper:
            for param in self.encoder.latent_encoder.parameters():
                param.requires_grad = False
            self.encoder.latent_encoder.eval()
            for param in self.decoder.latent_decoder.parameters():
                param.requires_grad = False
            self.decoder.latent_decoder.eval()
    
    def forward(self, x):

        state, latent_gt = self.encoder(x)

        output, latent = self.decoder(state)

        return output, latent, state, latent_gt