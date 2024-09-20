import yaml
import inspect
import torch
import numpy as np
import os
import json
import shutil
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from pytorch_lightning.tuner import Tuner
import plotly.graph_objects as go
from scipy import stats


def remove_outlier(x, percentile=98):
    if x.size == 0:
        return x
    else:
        thresh = np.percentile(x, percentile)
        return x[x <= thresh]


def scale_reject_ratio(pred_len, reject, reject_data):
    
    # calculate reject ratio

    reject_data = np.array(reject_data)
    
    reject_ratio_mean = np.mean(reject, axis=0)
    reject_ratio_sem = stats.sem(reject, axis=0)
    reject_ratio_data = np.mean(reject_data, axis=0)

    # rescale reject ratio using ground truth data reject ratio
    for p in range(pred_len):
        reject_ratio_mean[p] = (reject_ratio_mean[p] - reject_ratio_data[p]) / (1.0 - reject_ratio_data[p])
        reject_ratio_sem[p] = reject_ratio_sem[p] / (1.0 - reject_ratio_data[p])
    reject_ratio_mean = np.maximum.accumulate(reject_ratio_mean)

    return reject_ratio_mean

def calc_theta_diff(th1, th2):
    diff = np.abs(th2 - th1)
    diff = np.minimum(diff, 2*np.pi-diff)
    return diff

def get_experiment_dim_stats(dataset, model='encoder-decoder'):
    vars_dir = os.path.join('./outputs', dataset, 'variables')
    dims_all = []
    for seed in [1, 2, 3]:
        filepath = os.path.join(vars_dir, model+'_'+str(seed), 'intrinsic_dimension.npy')
        dims = np.load(filepath)
        dims_all.append(dims)
    dims_all = np.concatenate(dims_all)
    dim_mean = np.mean(dims_all)
    dim_std = np.std(dims_all)
    print('Mean (±std):', '%.2f (±%.2f)' % (dim_mean, dim_std))
    print('Confidence interval:', '(%.1f, %.1f)' % (dim_mean-1.96*dim_std, dim_mean+1.96*dim_std))

def get_experiment_dim(dataset, seed): 

    if dataset == 'single_pendulum':
        return 2
    if dataset == 'spring_mass':
        return 2
    if dataset == 'double_pendulum':
        return 4
    if dataset == 'fire':
        return 24
    if dataset == 'cylindrical_flow':
        return 3

def create_name(args, **kwargs):
    if 'regress' in args.model_name:
        name = [args.model_name, str(args.seed), args.nsv_model_name]
        if hasattr(args, "filter_data") and args.filter_data == True:
            name.append("filtered")
        model_name = '_'.join(name)
    elif 'smooth' in args.model_name:
        model_name = '_'.join([args.model_name, str(args.seed), args.reconstruct_loss_type, str(args.reconstruct_loss_weight), args.smooth_loss_type, str(args.smooth_loss_weight), args.regularize_loss_type, str(args.regularize_loss_weight), str(args.annealing)])
    else:
        model_name = '_'.join([args.model_name, str(args.seed)])

    print(model_name)
    return model_name


def tuple2name(file_tuple):

    return str(file_tuple[0].item()) + '/' + str(file_tuple[1].item()).zfill(2) + '.png' 

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def get_validArgs(cls, args):

    params = vars(args)
    valid_kwargs = inspect.signature(cls.__init__).parameters
    network_kwargs = {name: params[name] for name in valid_kwargs if name in params}

    return network_kwargs


def seed(cfg):
    torch.manual_seed(cfg.seed)

def remove_duplicates(X):
    return np.unique(X, axis=0)

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def mkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_weightPath(args, last):

    weight_dir = os.getcwd() + '/' + args.output_dir + '/' + args.dataset + "/checkpoints/" + create_name(args)
    print(weight_dir)

    if not os.path.exists(weight_dir) or len(os.listdir(weight_dir)) < 1:
        return None
    
    if last:
        return os.path.join(weight_dir, "last.ckpt")  
    else:
        items = os.listdir(weight_dir)

        for i in items:
            if "last" not in i:
                return os.path.join(weight_dir, i)
        
        return os.path.join(weight_dir, "last.ckpt") 

def analyze_pendulum_train_data(data_path, bins=90):

    phys_vars = np.load(os.path.join(data_path, 'phys_vars.npy'), allow_pickle=True).item()

    with open(os.path.join(data_path, 'datainfo', 'data_split_dict_1.json'), 'r') as file:
        seq_dict = json.load(file)

    for seq in ['train', 'test', 'val']:
        
        theta = phys_vars['theta']
        vel_theta = phys_vars['vel_theta']

        theta_seq = np.concatenate(theta[seq_dict[seq]])
        vel_theta_seq = np.concatenate(vel_theta[seq_dict[seq]])

        plt.figure()
        plt.hist(theta_seq, bins=bins, color='b')
        plt.xlabel('theta (from bottom)')
        plt.ylabel('# frames')
        plt.savefig(os.path.join(data_path, f'theta_{seq}_distribution.png'))

        plt.figure()
        plt.hist(vel_theta_seq, bins=bins, color='b')
        plt.xlabel('vel_theta')
        plt.ylabel('# frames')
        plt.savefig(os.path.join(data_path, f'vel_theta_{seq}_distribution.png'))

        plt.figure()
        plt.hist2d(theta_seq, vel_theta_seq, bins=(bins,bins))
        plt.xlabel('theta (from bottom)')
        plt.ylabel('vel_theta')
        plt.colorbar()
        plt.savefig(os.path.join(data_path, f'thetaXvel_theta_{seq}_distribution.png'))

        invert_theta_angles = (theta_seq + np.pi)%(2*np.pi)
        plt.figure()
        plt.hist(invert_theta_angles, bins=bins, color='r')
        plt.xlabel('theta (from top)')
        plt.ylabel('# frames')
        plt.savefig(os.path.join(data_path, f'theta_{seq}_distribution_invert.png'))

        plt.figure()
        plt.hist2d(invert_theta_angles, vel_theta_seq, bins=(bins,bins))
        plt.xlabel('theta (from top)')
        plt.ylabel('vel_theta')
        plt.colorbar()
        plt.savefig(os.path.join(data_path, f'thetaXvel_theta_{seq}_distribution_invert.png'))
        


        
    

        
