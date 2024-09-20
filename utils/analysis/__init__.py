import os
import sys
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from .eval_physics_single_pendulum import eval_phys_data_single_pendulum
from .eval_physics_double_pendulum import eval_phys_data_double_pendulum
from .eval_physics_spring_mass import eval_phys_data_spring_mass
from .eval_physics_cylindrical_flow import eval_phys_data_cylindrical_flow


class Physics_Evaluator:
    def __init__(self, dataset):

        self.dataset = dataset

        if self.dataset == 'cylindrical_flow':

            self.c2idx = torch.jit.load('./data/cylindrical_flow/c2idx.pt')
            self.c2idx.cuda()
            self.c2idx.eval()
    
    def eval_physics(self, data_filepath, num_vids, num_frms, save_path, return_marked=False):

        if self.dataset == "single_pendulum":
            return eval_phys_data_single_pendulum(data_filepath, num_vids, num_frms, save_path, return_marked=return_marked)
        elif self.dataset == "double_pendulum":
            return eval_phys_data_double_pendulum(data_filepath, num_vids, num_frms, save_path, return_marked=return_marked)
        elif self.dataset == "spring_mass":
            return eval_phys_data_spring_mass(data_filepath, num_vids, num_frms, save_path, return_marked=return_marked)
        elif self.dataset == "cylindrical_flow":
            self.c2idx.cuda()
            self.c2idx.eval()
            return eval_phys_data_cylindrical_flow(data_filepath, num_vids, num_frms, save_path, c2idx = self.c2idx, return_marked=return_marked)

    def get_phys_vars(self, include_reject=False):
        if self.dataset == "single_pendulum":
            if include_reject:
                return ['reject', 'theta', 'vel_theta', 'kinetic energy', 'potential energy', 'total energy']
            else:
                return ['theta', 'vel_theta', 'kinetic energy', 'potential energy', 'total energy']
        if self.dataset == "spring_mass":
            if include_reject:
                return ['reject', 'x', 'vel_x', 'kinetic energy', 'potential energy', 'total energy']
            else:
                return ['x', 'vel_x', 'kinetic energy', 'potential energy', 'total energy']
        elif self.dataset == "double_pendulum":
            if include_reject:
                return ['reject', 'theta_1', 'vel_theta_1', 'theta_2', 'vel_theta_2', 'kinetic energy', 'potential energy', 'total energy']
            else:
                return ['theta_1', 'vel_theta_1', 'theta_2', 'vel_theta_2', 'kinetic energy', 'potential energy', 'total energy']
        elif self.dataset == "cylindrical_flow":
            if include_reject:
                return ['reject', 'total energy']
            else:
                return ['total energy']
        else:
            return []


if __name__ == '__main__':
    dataset = str(sys.argv[1])
    data_filepath = '../data/' + dataset
    save_path = os.path.join(data_filepath, 'phys_vars.npy')

    evaluator = Physics_Evaluator()
    
    if dataset == 'single_pendulum':
        eval_phys_data_single_pendulum(data_filepath, 1200, 60, save_path)
    elif dataset == 'spring_mass':
        eval_phys_data_spring_mass(data_filepath, 1200, 60, save_path)
    elif dataset == 'double_pendulum':
        eval_phys_data_double_pendulum(data_filepath, 1100, 60, save_path)
    elif dataset == 'cylindrical_flow':
        eval_phys_data_cylindrical_flow(data_filepath, 1200, 200, save_path)
    else:
        assert False, 'Unknown system...'