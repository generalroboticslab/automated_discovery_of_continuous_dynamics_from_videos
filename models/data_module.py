import os
import sys
import json
import glob
import torch
import itertools
import numpy as np
from PIL import Image
from scipy import misc
import scipy.io as sio
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torchvision import datasets, transforms

import pytorch_lightning as pl
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from utils.misc import *
import collections


class RegressDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_filepath: str = "./data/",
                 output_dir: str = "outputs",
                 dataset: str = "single_pendulum",
                 nsv_model_name: str = "phys-simu",
                 decay_rate: float = 0.5,
                 train_batch: int = 32,
                 val_batch: int = 32,
                 test_batch: int = 32,
                 num_workers: int = 1,
                 seed: int = 1,
                 shuffle: bool=True,
                 pred_length: int=56,
                 filter_data: bool=False,
                 percentile: int=90,
                 data_annealing_list: list=[],
                 **kwargs) -> None:
        super().__init__()
        
        self.data_name = dataset
        self.nsv_model_name = nsv_model_name
        if 'phys' in self.nsv_model_name:
            self.data_path = data_filepath
        else:
            self.data_path = output_dir
        self.decay_rate = decay_rate
        
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.test_batch = test_batch
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pred_length = pred_length
        self.filter_data = filter_data
        self.percentile = percentile

        self.annealing_list = data_annealing_list

    def setup(self, stage: str):
        self.train_dataset = RegressDataset(data_filepath=self.data_path,
                                                    flag='train',
                                                    seed=self.seed,
                                                    object_name=self.data_name,
                                                    nsv_model_name=self.nsv_model_name,
                                                    decay_rate=self.decay_rate,
                                                    pred_length=self.pred_length,
                                                    filter_data=self.filter_data,
                                                    percentile=self.percentile)
        self.val_dataset = RegressDataset(data_filepath=self.data_path,
                                                    flag='val',
                                                    seed=self.seed,
                                                    object_name=self.data_name,
                                                    nsv_model_name=self.nsv_model_name,
                                                    decay_rate=self.decay_rate,
                                                    pred_length=self.pred_length,
                                                    filter_data=self.filter_data,
                                                    percentile=self.percentile)
        self.test_dataset = RegressDataset(data_filepath=self.data_path,
                                                    flag='test',
                                                    seed=self.seed,
                                                    object_name=self.data_name,
                                                    nsv_model_name=self.nsv_model_name,
                                                    decay_rate=self.decay_rate,
                                                    pred_length=self.pred_length,
                                                    filter_data=self.filter_data,
                                                    percentile=self.percentile)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.train_batch,
                                                   shuffle=self.shuffle,
                                                   pin_memory=True,
                                                   num_workers=self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                 batch_size=self.val_batch,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.test_batch,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_workers)


class RegressDataset(Dataset):
    def __init__(self, data_filepath, flag, seed, object_name, nsv_model_name, decay_rate, pred_length, filter_data=False, percentile=99, **kwargs):
        super().__init__()
        self.seed = seed
        self.flag = flag
        self.object_name = object_name
        self.data_filepath = data_filepath
        self.nsv_model_name = nsv_model_name
        self.decay_rate = decay_rate
        self.pred_length = pred_length
        self.filter_data = filter_data
        self.percentile = percentile
        
        self.get_states()

    
    def filter_states(self,):

        suf = '' if self.flag == 'test' else '_' + self.flag
        nsv_filepath = os.path.join(self.data_filepath, self.object_name, 'variables'+suf, self.nsv_model_name)
        
        trajectories = copy.deepcopy(self.states)

        finite_difference = []
        for vid_idx in trajectories.keys():
            trajectories[vid_idx] = np.array(trajectories[vid_idx])

            for i in range(1,trajectories[vid_idx].shape[0]):
                for j in range(trajectories[vid_idx].shape[1]):
                    if trajectories[vid_idx][i,j] - trajectories[vid_idx][i-1,j] > 1:
                        trajectories[vid_idx][i:,j] -= 2
                    elif trajectories[vid_idx][i,j] - trajectories[vid_idx][i-1,j] < -1:
                        trajectories[vid_idx][i:,j] += 2
                
            finite_difference.extend(trajectories[vid_idx][1:,:]-trajectories[vid_idx][:-1,:])

        finite_difference = np.array(finite_difference)
        output_norm = np.percentile(np.abs(finite_difference), self.percentile, axis=0)
        
        print("Filtering States (Keeping {}th percentile)".format(self.percentile))

        print("Finite Difference Limit: ", output_norm)

        print("Num traj before: ", len(trajectories.keys()))

        invalid = []

        for vid_idx, seq in trajectories.items():

            fd = np.abs(seq[1:,:] - seq[:-1,:])
            
            for i in range(seq.shape[-1]):
                if np.any(fd[:,i] > output_norm[i]):
                    invalid.append(vid_idx)
                    break
        
        np.save(os.path.join(nsv_filepath, "total.npy"), np.array(list(self.states.keys())))
        np.save(os.path.join(nsv_filepath, "invalid.npy"), np.array(invalid))

        if not len(invalid) == len(self.states.keys()):
            for id in invalid:
                del self.states[id]

        print("Num traj after: ", len(self.states.keys()))

        return invalid

    def get_all_filelist(self):
        vid_list = list(self.states.keys())

        filelist = []
        for vid_idx in vid_list:
            num_frames = self.states[vid_idx].shape[0]
            for p_frame in range(num_frames - 1):
                par_list = (vid_idx, p_frame)
                filelist.append(par_list)

        return filelist

    def get_states(self):
        suf = '' if self.flag == 'test' else '_' + self.flag

        nsv_filepath = os.path.join(self.data_filepath, self.object_name, 'variables'+suf, self.nsv_model_name)
        ids = np.load(os.path.join(nsv_filepath, 'ids.npy'))
        nsv = np.load(os.path.join(nsv_filepath, 'refine_latent.npy'))

        self.states = self.trajectories_from_data_ids(ids, nsv)

        if self.filter_data:
            print("Filtering {}th percentile".format(self.percentile))
            self.filter_states()
        
        self.all_filelist = self.get_all_filelist()


    def __len__(self):
        return len(self.all_filelist)

    def __getitem__(self, idx):
        vid_idx, p_frame = self.all_filelist[idx]
        data, target, weight = self.get_data(vid_idx, p_frame)
        
        file_tuple = torch.tensor([vid_idx , p_frame])
        return data, target, weight, file_tuple

    def get_data(self, vid_idx, p_frame):

        rho = self.decay_rate
        seq = self.states[vid_idx]

        data_nsv = torch.tensor(seq[p_frame])

        target_nsv = torch.tensor(seq[p_frame + 1:])

        cur_step = data_nsv

        for i in range(target_nsv.shape[0]):
            for j in range(target_nsv.shape[1]):
                if target_nsv[i,j] - cur_step[j] > 1:
                    target_nsv[i:,j] -= 2
                elif target_nsv[i,j] - cur_step[j] < -1:
                    target_nsv[i:,j] += 2
            cur_step = target_nsv[i]


        target_nsv = torch.cat((target_nsv, torch.zeros(p_frame, seq.shape[1])), 0)

        weight = rho ** torch.arange(seq.shape[0]-p_frame-1)
        weight = torch.reshape(weight, (-1, 1))
        weight = torch.cat((weight, torch.zeros(p_frame, 1)), 0).float()

        weight = weight / (weight.sum() + .0000001)

        return data_nsv, target_nsv, weight

    def trajectories_from_data_ids(self, ids, nsv):


        id2index = {tuple(id): i for i, id in enumerate(ids)}
        trajectories = collections.defaultdict(list)
        indices = collections.defaultdict(list)

        for id in sorted(id2index.keys()):
            i = id2index[id]
            trajectories[id[0]].append(nsv[i])
            indices[id[0]].append(ids[i][1])

        for vid_idx in trajectories.keys():
            #print("Before: ", trajectories[vid_idx])
            #print(indices[vid_idx])
            #print(trajectories[vid_idx])
            traj = [x for _, x in sorted(zip(indices[vid_idx], trajectories[vid_idx]))]
            #print((traj == trajectories[vid_idx]))
            trajectories[vid_idx] = np.array(traj)

            # if self.flag == 'train':
            #     ipdb.set_trace()
            #print(trajectories[vid_idx])

        return trajectories


class SimulationDataModule(pl.LightningDataModule):
    def __init__(self, data_filepath: str = "./data/", dataset: str = "single_pendulum", model_name: str = 'encoder-decoder-64', train_batch: int = 32, val_batch: int = 32, test_batch: int = 32, num_workers=32, seed = 1, shuffle=True, data_annealing_list: list=[], **kwargs):
        super().__init__()
        
        self.data_path = data_filepath
        self.data_name = dataset
        self.model_name = model_name
        
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.test_batch = test_batch
        self.num_workers = num_workers
        self.seed = seed

        self.shuffle = shuffle
        self.annealing_list = data_annealing_list

    def setup(self, stage: str):
        if 'smooth' in self.model_name:

            self.train_dataset = NeuralPhysSmoothDataset(data_filepath=self.data_path,
                                                        flag='train',
                                                        seed=self.seed,
                                                        object_name=self.data_name)
            self.val_dataset = NeuralPhysSmoothDataset(data_filepath=self.data_path,
                                                        flag='val',
                                                        seed=self.seed,
                                                        object_name=self.data_name)
            self.test_dataset = NeuralPhysSmoothDataset(data_filepath=self.data_path,
                                                        flag='test',
                                                        seed=self.seed,
                                                        object_name=self.data_name)
        
        else:

            self.train_dataset = NeuralPhysDataset(data_filepath=self.data_path,
                                                        flag='train',
                                                        seed=self.seed,
                                                        object_name=self.data_name)
            self.val_dataset = NeuralPhysDataset(data_filepath=self.data_path,
                                                        flag='val',
                                                        seed=self.seed,
                                                        object_name=self.data_name)
            self.test_dataset = NeuralPhysDataset(data_filepath=self.data_path,
                                                        flag='test',
                                                        seed=self.seed,
                                                        object_name=self.data_name)
            

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.train_batch,
                                                   shuffle=self.shuffle,
                                                   pin_memory=True,
                                                   num_workers=self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                 batch_size=self.val_batch,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.test_batch,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_workers)


class NeuralPhysSmoothDataset(Dataset):
    def __init__(self, data_filepath, flag, seed, object_name="double_pendulum"):
        self.seed = seed
        self.flag = flag
        self.object_name = object_name
        self.data_filepath = data_filepath
        self.all_filelist = self.get_all_filelist()

    def get_all_filelist(self):
        filelist = []
        obj_filepath = os.path.join(self.data_filepath, self.object_name)
        # get the video ids based on training or testing data
        with open(os.path.join(self.data_filepath, self.object_name, 'datainfo', f'data_split_dict_{self.seed}.json'), 'r') as file:
            seq_dict = json.load(file)
        vid_list = seq_dict[self.flag]

        # go through all the selected videos
        for vid_idx in vid_list:
            seq_filepath = os.path.join(obj_filepath, str(vid_idx))
            num_frames = len(os.listdir(seq_filepath))
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            for p_frame in range(num_frames - 3):
                par_list = {'seq_filepath':seq_filepath, 'suf':suf, 'p_frame':p_frame}
                filelist.append(par_list)
        return filelist

    def __len__(self):
        return len(self.all_filelist)

    def __getitem__(self, idx):
        par_list = self.all_filelist[idx]
        seq_filepath, suf = par_list['seq_filepath'], par_list['suf']
        p_frame = par_list['p_frame']
        # t, t+1
        data = []
        for p in range(2):
            data.append(self.get_data(os.path.join(seq_filepath, str(p_frame + p) + '.' + suf)))
        data = torch.cat(data, 2)
        # t+2, t+3
        target = []
        for p in range(2, 4):
            target.append(self.get_data(os.path.join(seq_filepath, str(p_frame + p) + '.' + suf)))
        target = torch.cat(target, 2)
        #t+1, t+2
        in_between = []
        in_between.append(self.get_data(os.path.join(seq_filepath, str(p_frame + 1) + '.' + suf)))
        in_between.append(self.get_data(os.path.join(seq_filepath, str(p_frame + 2) + '.' + suf)))
        in_between = torch.cat(in_between, 2)

        file_tuple = torch.tensor([eval(seq_filepath.split('/')[-1]) , p_frame])
        
        return data, target, in_between, file_tuple

    def get_data(self, filepath):
        data = Image.open(filepath)
        data = data.resize((128, 128))
        data = np.array(data)
        data = torch.tensor(data / 255.0)
        data = data.permute(2, 0, 1).float()
        return data


class NeuralPhysDataset(Dataset):
    def __init__(self, data_filepath, flag, seed, object_name="double_pendulum"):
        super().__init__()
        self.seed = seed
        self.flag = flag
        self.object_name = object_name
        self.data_filepath = data_filepath
        self.all_filelist = self.get_all_filelist()

    def get_all_filelist(self):
        filelist = []
        obj_filepath = os.path.join(self.data_filepath, self.object_name)
        # get the video ids based on training or testing data
        with open(os.path.join(self.data_filepath, self.object_name, 'datainfo', f'data_split_dict_{self.seed}.json'), 'r') as file:
            seq_dict = json.load(file)
        vid_list = seq_dict[self.flag]

        # go through all the selected videos and get the triplets: input(t, t+1), output(t+2)
        for vid_idx in vid_list:
            seq_filepath = os.path.join(obj_filepath, str(vid_idx))
            num_frames = len(os.listdir(seq_filepath))
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            for p_frame in range(num_frames - 3):
                par_list = []
                for p in range(4):
                    par_list.append(os.path.join(seq_filepath, str(p_frame + p) + '.' + suf))
                filelist.append(par_list)
        return filelist

    def __len__(self):
        return len(self.all_filelist)

    # 0, 1 -> 2, 3
    def __getitem__(self, idx):
        par_list = self.all_filelist[idx]
        data = []
        for i in range(2):
            data.append(self.get_data(par_list[i])) # 0, 1
        data = torch.cat(data, 2)
        target = []
        target.append(self.get_data(par_list[-2])) # 2
        target.append(self.get_data(par_list[-1])) # 3
        target = torch.cat(target, 2)

        file_tuple = torch.tensor([eval(par_list[0].split('/')[-2]) , eval(par_list[0].split('/')[-1][:-4])])

        return data, target, file_tuple

    def get_data(self, filepath):
        data = Image.open(filepath)
        data = data.resize((128, 128))
        data = np.array(data)
        data = torch.tensor(data / 255.0)
        data = data.permute(2, 0, 1).float()
        return data