import time
import os
import numpy as np
import collections
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from tqdm import tqdm
from pysr import PySRRegressor
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from torchvision.utils import save_image
from PIL import Image
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import sympy as sp
from inspect import signature

import pytorch_lightning as pl

from torchvision import transforms
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from utils.intrinsic_dimension_estimation import ID_Estimator
from utils.misc import *
from utils.show import *
from utils.analysis import Physics_Evaluator
from models.sub_modules import computeJacobian


def frange_cycle_linear(n_iter, full_len, pretrain_period, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(full_len) * stop
    L[:pretrain_period] = start  # Setting the pretraining period coefficients to start
    
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period + pretrain_period) < n_iter + pretrain_period):
            L[int(i + c * period + pretrain_period)] = v
            v += step
            i += 1
            
    return L  

class CyclicalAnnealingCallback(pl.Callback):
    def __init__(self, 
                 epochs, 
                 pretrain_epochs, 
                 anneal_epochs=None,
                 annealing=False,
                 **kwargs):
        """
        Implements cyclical annealing for the loss weights and discount factor

        Args:

        epochs: maximum number of epochs to train for
        pretrain_epochs: number of epochs to pretrain for
        """

        self.epochs = epochs
        self.anneal_epochs = anneal_epochs if anneal_epochs else epochs
        self.annealing = annealing
        self.pretrain_period = pretrain_epochs
        self.annealing = annealing
        self.schedules = {}
        self.n_iter = self.anneal_epochs - pretrain_epochs
    
    def add_annealing(self, location, value, start, stop, n_cycle, ratio, type='linear'):

        if type == 'linear':
            self.schedules[value] = (location, frange_cycle_linear(self.n_iter, self.epochs, self.pretrain_period, start, stop, n_cycle, ratio), stop)
        elif type == 'sigmoid':
            self.schedules[value] = (location, frange_cycle_linear(self.n_iter, self.epochs, self.pretrain_period, start, stop, n_cycle, ratio), stop)
        else:
            raise ValueError("type must be either 'linear' or 'sigmoid'")
    
    def on_train_start(self, trainer, pl_module):
        
        for item in pl_module.annealing_list:
            value, start, stop, n_cycle, ratio, type = item[0], item[1], item[2], item[3], item[4], item[5]

            self.add_annealing("pl_module", value, start, stop, n_cycle, ratio, type)
        
        for item in trainer.datamodule.annealing_list:
            value, start, stop, n_cycle, ratio, type = item[0], item[1], item[2], item[3], item[4], item[5]

            self.add_annealing("dataset", value, start, stop, n_cycle, ratio, type)

    def on_train_epoch_start(self, trainer, pl_module):

        for value, schedule in self.schedules.items():
            if self.annealing:
                if schedule[0] == 'pl_module':
                    updated_value = schedule[1][trainer.current_epoch]
                    pl_module.__dict__[value] = updated_value
                elif schedule[0] == 'dataset':
                    updated_value = schedule[1][trainer.current_epoch]
                    trainer.datamodule.train_dataset.__dict__[value] = updated_value
            else:
                if schedule[0] == 'pl_module':
                    updated_value = schedule[2]
                    pl_module.__dict__[value] = updated_value
                elif schedule[0] == 'dataset':
                    updated_value = schedule[2]
                    trainer.datamodule.train_dataset.__dict__[value] = updated_value

            pl_module.log(value, updated_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def on_validation_epoch_start(self, trainer, pl_module):

        for value, schedule in self.schedules.items():
            if schedule[0] == 'pl_module':
                updated_value = schedule[2]
                pl_module.__dict__[value] = updated_value
            elif schedule[0] == 'dataset':
                updated_value = schedule[2]
                trainer.datamodule.train_dataset.__dict__[value] = updated_value

            pl_module.log("val_"+value, updated_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

class LitProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)

        items.pop("v_num", None)

        return items

class IntrinsicDimensionEstimator(Callback):

    def eval_id_latent(self, vars_filepath, if_all_methods):

        latent = np.load(os.path.join(vars_filepath, 'latent.npy'))

        print(f'Number of samples: {latent.shape[0]}; Latent dimension: {latent.shape[1]}')

        id = np.load(os.path.join(vars_filepath, 'ids.npy'))

        latent = remove_duplicates(latent)
        print(f'Number of samples (duplicates removed): {latent.shape[0]}')
        
        estimator = ID_Estimator()
        k_list = (latent.shape[0] * np.linspace(0.008, 0.016, 5)).astype('int')
        print(f'List of numbers of nearest neighbors: {k_list}')
        if if_all_methods:
            dims = estimator.fit_all_methods(latent, k_list)
            np.save(os.path.join(vars_filepath, 'intrinsic_dimension_all_methods.npy'), dims)
        else:
            dims = estimator.fit(latent, k_list)
            np.save(os.path.join(vars_filepath, 'intrinsic_dimension.npy'), dims)
        
        return dims

    def on_test_end(self, trainer, pl_module):

        np.save(os.path.join(pl_module.var_log_dir, 'ids.npy'), pl_module.all_filepaths)
        np.save(os.path.join(pl_module.var_log_dir, 'latent.npy'), pl_module.all_latents)

        if 'train' in pl_module.var_log_dir or "val" in pl_module.var_log_dir:
            return

        dims = self.eval_id_latent(pl_module.var_log_dir, False)

class SmoothnessEvaluator(Callback):

    def __init__(self,):

        super().__init__()
        self.window = 10

    def trajectories_from_data_ids(self, ids, nsv):
        id2index = {tuple(id): i for i, id in enumerate(ids)}
        trajectories = collections.defaultdict(list)

        for id in sorted(id2index.keys()):
            i = id2index[id]
            trajectories[id[0]].append(nsv[i])

        return trajectories

    def physical_variables_from_data_ids(self, phys_all, ids):
        phys_vars_list = []
        for p_var in phys_all.keys():
            if p_var == 'reject':
                continue
            for t in range(4):
                phys_vars_list.append(f'{p_var} (t={t})')

        num_data = len(ids)
        phys = {p_var:np.zeros(num_data) for p_var in phys_vars_list}

        for n in range(num_data):
            vid_n, frm_n = ids[n]
            for p_var in phys_all.keys():
                if p_var == 'reject':
                    continue
                for t in range(4):
                    phys[f'{p_var} (t={t})'][n] = phys_all[p_var][vid_n, frm_n+t]

        return phys
    

    def eval_smooth(self, trajectories, task_log_dir, invalid_trajectories, dt, data_min, data_max):
        variation_mean = {}
        variation_ord2_mean = {}
        deviation = {}

        variation_max = {}
        variation_ord2_max = {}

        tangling_mean = {}
        tangling_max = {}

        spline_fit_save_path = os.path.join(task_log_dir, 'spline_fitting')
        mkdir(spline_fit_save_path)
        mkdir(os.path.join(task_log_dir, 'nsv_trajectories'))
        mkdir(os.path.join(task_log_dir, "time_series"))
        mkdir(os.path.join(task_log_dir, "first_order_derivatives"))

        np.save(os.path.join(task_log_dir, "invalid.npy"), np.array(invalid_trajectories))

        print('Evaluating Smoothness')
        for p, traj in tqdm(trajectories.items()):
            traj_arr = np.array(traj)
            tangling_mean[p], tangling_max[p] = self.calculate_tangling_mean_max(traj_arr, dt)
            variation_mean[p], variation_ord2_mean[p] = self.calculate_variation_mean(traj_arr, dt)
            deviation[p] = self.calculate_deviation(traj_arr, dt)
            variation_max[p], variation_ord2_max[p] = self.calculate_variation_max(traj_arr, dt)
            self.visualize_trajectory(traj_arr, task_log_dir, p, dt, data_min, data_max)
        
        variation_mean_arr = np.array(list(variation_mean.values()))
        variation_ord2_mean_arr = np.array(list(variation_ord2_mean.values()))
        np.save(os.path.join(spline_fit_save_path, 'pre_filter_variation_mean.npy'), variation_mean_arr)
        np.save(os.path.join(spline_fit_save_path, 'pre_filter_variation_ord2_mean.npy'), variation_ord2_mean_arr)
        variation_max_arr = np.array(list(variation_max.values()))
        variation_ord2_max_arr = np.array(list(variation_ord2_max.values()))
        np.save(os.path.join(spline_fit_save_path, 'pre_filter_variation_max.npy'), variation_max_arr)
        np.save(os.path.join(spline_fit_save_path, 'pre_filter_variation_ord2_max.npy'), variation_ord2_max_arr)

        pre_filter_deviation_arr = np.array(list(deviation.values()))
        post_filter_deviation_arr = np.array(list(deviation[traj] for traj in deviation.keys() if traj not in invalid_trajectories))
        np.save(os.path.join(spline_fit_save_path, 'pre_filter_deviation.npy'), pre_filter_deviation_arr)
        np.save(os.path.join(spline_fit_save_path, 'post_filter_deviation.npy'), post_filter_deviation_arr)

        pre_filter_tangling_arr = np.array(list(tangling_mean.values()))
        pre_filter_tangling_max_arr = np.array(list(tangling_max.values()))
        post_filter_tangling_arr =  np.array(list(tangling_mean[traj] for traj in deviation.keys() if traj not in invalid_trajectories))
        post_filter_tangling_max_arr =  np.array(list(tangling_max[traj] for traj in deviation.keys() if traj not in invalid_trajectories))
        np.save(os.path.join(spline_fit_save_path, 'pre_filter_tangling.npy'), pre_filter_tangling_arr)
        np.save(os.path.join(spline_fit_save_path, 'pre_filter_tangling_max.npy'), pre_filter_tangling_max_arr)
        np.save(os.path.join(spline_fit_save_path, 'post_filter_tangling.npy'), post_filter_tangling_arr)
        np.save(os.path.join(spline_fit_save_path, 'post_filter_tangling_max.npy'), post_filter_tangling_max_arr)

        print('pre filter mean variation:', '%.2f (±%.2f)' % (np.mean(variation_mean_arr), np.std(variation_mean_arr)))
        print('pre filter mean second order variation:', '%.2f (±%.2f)' % (np.mean(variation_ord2_mean_arr), np.std(variation_ord2_mean_arr)))

        print('pre filter max max variation:', '%.2f' % (np.max(variation_max_arr)))
        print('pre filter max max second order variation:', '%.2f' % (np.max(variation_ord2_max_arr)))

        print('pre filter mean max variation:', '%.2f' % (np.mean(variation_max_arr)))
        print('pre filter mean max second order variation:', '%.2f' % (np.mean(variation_ord2_max_arr)))

        print('pre filter trajectory deviation mean:', '%.2f' % (np.mean(pre_filter_deviation_arr)))

        print('pre filter tangling mean:', '%.2f' % (np.mean(pre_filter_tangling_arr)))
        print('pre filter tangling max:', '%.2f' % (np.max(pre_filter_tangling_max_arr)))

    def get_smooth_trajectory(self, traj, dt, window):

        smooth_traj = traj.copy()

        t = np.arange(traj.shape[0]) * dt
        cs = CubicSpline(t[::window], traj[::window,:])

        smooth_traj = cs(t)

        return smooth_traj

    def calculate_tangling_mean_max(self, traj, dt):

        traj_len = traj.shape[0]

        t = np.arange(traj_len) * dt
        cs = CubicSpline(t, traj)
        d_traj = cs(t, 1)

        x = traj.copy()
        dx = np.array(d_traj)

        x = np.reshape(x, (1, traj_len, x.shape[1]))
        dx = np.reshape(dx, (1, traj_len, dx.shape[1]))

        target_x = np.broadcast_to(x, (traj_len, traj_len, x.shape[2]))
        target_dx = np.broadcast_to(dx, (traj_len, traj_len, dx.shape[2]))

        x = np.reshape(x, (traj_len, 1, x.shape[2]))
        dx = np.reshape(dx, (traj_len, 1, dx.shape[2]))

        diff_x = np.sum((target_x-x)**2, axis=2)
        diff_dx = np.sum((target_dx-dx)**2, axis=2)

        fraction = diff_dx / (diff_x + np.finfo(np.float32).eps)

        q = np.max(fraction, axis=1)

        return np.mean(q), np.max(q)

    def calculate_deviation(self, traj, dt):

        smooth_traj = self.get_smooth_trajectory(traj, dt=dt, window=self.window)

        deviation = np.sum(np.abs(smooth_traj - traj))*dt

        return deviation

    def calculate_variation_mean(self, traj, dt):
        
        t = np.arange(traj.shape[0]) * dt
        cs = CubicSpline(t, traj)
        d2_traj = cs(t, 2)

        # Normalize by range
        rng = traj.max()-traj.min()

        # Finite Difference
        d_traj = (traj[1:] - traj[:-1]) / dt

        var_mean = dt * np.linalg.norm(d_traj, axis=1).sum() / rng
        var2_mean = dt * np.linalg.norm(d2_traj, axis=1).sum() / rng

        return var_mean, var2_mean

    def calculate_variation_max(self, traj, dt):

        # use cubic spline 
        t = np.arange(traj.shape[0]) * dt
        cs = CubicSpline(t, traj)
        d2_traj = cs(t, 2)

        # Normalize by range
        rng = traj.max()-traj.min()

        d_traj = (traj[1:] - traj[:-1]) / dt

        var_max = np.linalg.norm(d_traj, axis=1).max() / rng
        var2_max = np.linalg.norm(d2_traj, axis=1).max() / rng

        return var_max, var2_max
    

    def visualize_trajectory(self, traj, save_path, vid_idx, dt, data_min, data_max):

        num_components = min(traj.shape[-1],4)
        steps = traj.shape[0]
        
        data_full_max = np.stack([np.abs(data_min), data_max], axis=0).max(axis=0)

        if num_components == 4:
            fig = make_subplots(rows=num_components-1, cols=num_components-1)
        else:
            fig = go.Figure()

        if num_components > 3:
            for i in range(num_components-1):
                for j in range(i+1, num_components):
                    fig.add_trace(go.Scatter(x=traj[:,i], y=traj[:,j], mode='lines+markers', 
                                line=dict(color='rgba(0, 0, 0, 0.5)', width=3), marker=dict(size=4, color=np.arange(traj.shape[0]) * dt, colorscale='deep')), row=j, col=i+1)
                    fig.update_xaxes(row=j, col=i+1, range=[-1.1*data_full_max[i], 1.1*data_full_max[i]], tickmode='linear', tick0 = int(10*data_full_max[i])/10, dtick=int(10*data_full_max[i])/10)
                    fig.update_yaxes(row=j, col=i+1, range=[-1.1*data_full_max[j], 1.1*data_full_max[j]], tickmode='linear', tick0 = int(10*data_full_max[j])/10, dtick=int(10*data_full_max[j])/10)

                    if i == 0:
                        fig.update_yaxes(title_text=f"<b>V{j+1}</b>",row=j, col=i+1)
                    if j == num_components - 1:
                        fig.update_xaxes(title_text=f"<b>V{i+1}</b>",row=j, col=i+1)

            fig.update_layout(showlegend=False)
            update_figure_small(fig, True)  
        elif num_components == 3:
            fig.add_trace(go.Scatter3d(x=traj[:,0], y=traj[:,1], z=traj[:,2], mode='lines+markers', 
                        line=dict(color='rgba(0, 0, 0, 0.5)', width=3), marker=dict(size=2, color=np.arange(traj.shape[0]) * dt, colorscale='deep')
                    ))
            
            fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
                            xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                            yaxis=dict(title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                            zaxis=dict(title='<b>V3</b>', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10)))
            

            update_figure_3d(fig)

        else:

            traj = traj[0::2,:]
            dt = 2*dt
            steps = traj.shape[0]

            fig.add_trace(go.Scatter(x=traj[:,0], y=traj[:,1], mode='lines+markers',
                        line=dict(color='rgba(0, 0, 0, 0.5)', width=7), marker=dict(size=30, symbol="arrow", angleref="previous", color=np.arange(traj.shape[0]) * dt, colorscale='deep')))
            
            fig.update_layout(width=700, height=660,
                        xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                        yaxis=dict(title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                        showlegend=False)
            update_figure(fig, True)
                
        
        data_max_full = data_full_max.max()

        fig.write_image(os.path.join(save_path, f'nsv_trajectories/{vid_idx}.png'), scale=4)

        fig = go.Figure()
        for i in range(num_components):
            fig.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=traj[:,i], mode='lines', line=dict(color=cols[i], width=7), name=f"<b>V{i+1}</b>"))
        fig.update_layout(xaxis=dict(title='<b>Time (s)</b>', range=[0,dt*(steps+3)], nticks=3),
                            yaxis=dict(title='<b>V</b>', range=[-1.1*data_max_full, 1.1*data_max_full], tickmode='linear', tick0 = int(10*data_max_full)/10, dtick=int(10*data_max_full)/10))
        update_figure(fig, True)
        fig.write_image(os.path.join(save_path, 'time_series/'+str(vid_idx)+'.png'), scale=4)

        t = np.arange(traj.shape[0]) * dt
        cs = CubicSpline(t, traj)
        d_traj = cs(t, 1)
        
        fig = go.Figure()
        for i in range(num_components):
            fig.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-2), steps-1), y=d_traj[:,i], mode='lines', line=dict(color=cols[i], width=7), name=f"<b>dV{i+1}/dt</b>"))
        fig.update_layout(xaxis=dict(title='<b>Time (s)</b>', range=[0,dt*(steps+3)], nticks=3),
                            yaxis=dict(title='<b>dV/dt</b>', nticks=5))
        update_figure(fig, True)
        fig.write_image(os.path.join(save_path, 'first_order_derivatives/'+str(vid_idx)+'.png'), scale=4)


    def visualize_nsv_embedding(self, phys_all, ids_test, nsv_test, task_log_dir, data_max, data_min):
        embedding_save_path = os.path.join(task_log_dir, 'nsv_embedding')
        mkdir(embedding_save_path)

        phys_test = self.physical_variables_from_data_ids(phys_all, ids_test)
        phys_vars_list = phys_test.keys()
        nsv_test_arr = np.array(nsv_test)
        num_components = nsv_test_arr.shape[1]

        print('Visualizing NSV Embeddings')
        for p_var in tqdm(phys_vars_list):
            self.visualize_nsv(nsv_test_arr, phys_test[p_var], p_var, embedding_save_path, data_max, data_min)


    def visualize_nsv(self, data, v, v_name, save_path, data_max, data_min):
        num_components = data.shape[-1]

        data_full_max = np.stack([np.abs(data_min), data_max], axis=0).max(axis=0)

        if num_components != 2:

            fig = make_subplots(rows=num_components-1, cols=num_components-1)

            for i in range(num_components-1):
                for j in range(i+1, num_components):
                    fig.add_trace(go.Scatter(x=data[:,i], y=data[:,j], mode='markers',
                                marker=dict(size=4, color=v, colorbar=dict(title='',tickprefix="<b>",ticksuffix ="</b>", nticks=6),colorscale=colorscale)), row=j, col=i+1)
                    fig.update_xaxes(row=j, col=i+1, range=[-1.1*data_full_max[i], 1.1*data_full_max[i]], tickmode='linear', tick0 = int(10*data_full_max[i])/10, dtick=int(10*data_full_max[i])/10)
                    fig.update_yaxes(row=j, col=i+1, range=[-1.1*data_full_max[j], 1.1*data_full_max[j]], tickmode='linear', tick0 = int(10*data_full_max[j])/10, dtick=int(10*data_full_max[j])/10)


                    if i == 0:
                        fig.update_yaxes(title_text=f"<b>V{j+1}</b>",row=j, col=i+1)
                    if j == num_components - 1:
                        fig.update_xaxes(title_text=f"<b>V{i+1}</b>",row=j, col=i+1)


            fig.update_layout(showlegend=False)
            update_figure_small(fig, True)
        else:
            fig = go.Figure(data=go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers',marker=dict(size=8, color=v, colorbar=dict(title='',tickprefix="<b>",ticksuffix ="</b>", nticks=6),colorscale=colorscale)))
            
            fig.update_layout(xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                        yaxis=dict(scaleanchor="x", scaleratio=1, title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                        showlegend=False)
            update_figure(fig, True)
        fig.write_image(os.path.join(save_path, v_name+'.png'), scale=4)


    def find_filtered_trajectories(self, trajectories, percentile=99):

        finite_difference = []
        for vid_idx in trajectories.keys():
            trajectories[vid_idx] = np.array(trajectories[vid_idx])
            finite_difference.extend(trajectories[vid_idx][1:,:]-trajectories[vid_idx][:-1,:])

        finite_difference = np.array(finite_difference)
        output_norm = np.percentile(np.abs(finite_difference), percentile, axis=0)
        
        print("Filtering States (Keeping {}th percentile)".format(percentile))

        print("Finite Difference Limit: ", output_norm)

        print("Num traj before: ", len(trajectories.keys()))

        invalid = []

        for vid_idx, seq in trajectories.items():

            fd = np.abs(seq[1:,:] - seq[:-1,:])
            
            for i in range(seq.shape[-1]):
                if np.any(fd[:,i] > output_norm[i]):
                    invalid.append(vid_idx)
                    break
        
        print("Num traj after: ", len(trajectories.keys()) - len(invalid))

        if len(trajectories.keys()) - len(invalid) == 0:
            print("Refiltering using euclidean distance")

            output_norm = np.percentile((finite_difference**2).sum(axis=1), percentile)
            
            print("Finite Difference Limit: ", output_norm)

            invalid = []

            for vid_idx, seq in trajectories.items():

                fd = ((seq[1:,:] - seq[:-1,:])**2).sum(axis=1)
                
                if np.any(fd > output_norm):
                    invalid.append(vid_idx)
                    break
            
            print("Num traj after: ", len(trajectories.keys()) - len(invalid))

        return invalid


    def on_test_end(self, trainer, pl_module):

        # save variables
        np.save(os.path.join(pl_module.var_log_dir, 'ids.npy'), pl_module.all_filepaths)
        np.save(os.path.join(pl_module.var_log_dir, 'latent.npy'), pl_module.all_latents)
        np.save(os.path.join(pl_module.var_log_dir, 'refine_latent.npy'), pl_module.all_refine_latents)
        np.save(os.path.join(pl_module.var_log_dir, 'reconstructed_latent.npy'), pl_module.all_reconstructed_latents)

        if pl_module.test_mode == "save_train_data":
            return
        

        task_log_dir = os.path.join(pl_module.output_dir, pl_module.model.dataset, pl_module.task_log_dir, pl_module.model.name)
        mkdir(task_log_dir)

        ids_test = np.array(pl_module.all_filepaths)
        nsv_test = np.array(pl_module.all_refine_latents)
        
        trajectories_test = self.trajectories_from_data_ids(ids_test, nsv_test)
        invalid_trajectories = self.find_filtered_trajectories(trajectories_test)

        data_max = np.max(nsv_test, axis=0)
        data_min = np.min(nsv_test, axis=0)
    
        self.eval_smooth(trajectories_test, task_log_dir, invalid_trajectories, pl_module.dt, data_min, data_max)

        data_file_path = os.path.join('./data', pl_module.model.dataset)
        physics_variable_path = os.path.join(data_file_path, 'phys_vars.npy')
    
        phys_all = np.load(physics_variable_path, allow_pickle=True).item()

        self.visualize_nsv_embedding(phys_all, ids_test, nsv_test, task_log_dir, data_max, data_min)



class RegressEvaluator(Callback):

    def __init__(self,):

        super().__init__()
    
    def plot_trajectory(self, num_components, target_traj, pred_traj, save_path, traj_id, dt, data_max, data_min):

        num_components = min(pred_traj.shape[1], 4)
        steps = pred_traj.shape[0]

        data_full_max = np.stack([np.abs(data_min), data_max], axis=0).max(axis=0)
        data_max_full = data_full_max.max()

        fig = go.Figure()
        for i in range(num_components):
            fig.add_trace(go.Scatter(x=np.linspace(0,dt*steps, steps), y=pred_traj[:,i], line=dict(color=cols[i], width=7), name=f"<b>V{i+1}</b>"))
            fig.add_trace(go.Scatter(x=np.linspace(0,dt*steps, steps), y=target_traj[:,i], line=dict(color=cols[i], width=7, dash='dot'), name=f"V{i+1} ground truth"))
        fig.update_layout(xaxis=dict(title='<b>Time (s)</b>', range=[0,dt*(steps+3)], nticks=3),
                            yaxis=dict(title='<b>V</b>', range=[-1.1*data_max_full, 1.1*data_max_full], tickmode='linear', tick0 = int(10*data_max_full)/10, dtick=int(10*data_max_full)/10))
        update_figure(fig, True)
        fig.write_image(os.path.join(save_path, 'time_series/'+str(traj_id)+'.png'), scale=4)

        if num_components == 4:
            fig = make_subplots(rows=num_components-1, cols=num_components-1)
            x_max = np.max(np.abs(target_traj), axis=0)
            for i in range(num_components-1):
                for j in range(i+1, num_components):
                    fig.add_trace(go.Scatter(x=pred_traj[:,i], y=pred_traj[:,j], name='<b>prediction</b>',
                                    mode='lines', line=dict(width=1, color=cols[0])), row=j, col=i+1)
                    fig.add_trace(go.Scatter(x=target_traj[:,i], y=target_traj[:,j], name='<b>ground truth</b>',
                                    mode='lines', line=dict(width=0.75, color=cols[1], dash='dot')), row=j, col=i+1)
                    fig.update_xaxes(row=j, col=i+1, range=[-1.1*data_full_max[i], 1.1*data_full_max[i]], tickmode='linear', tick0 = int(10*data_full_max[i])/10, dtick=int(10*data_full_max[i])/10)
                    fig.update_yaxes(row=j, col=i+1, range=[-1.1*data_full_max[j], 1.1*data_full_max[j]], tickmode='linear', tick0 = int(10*data_full_max[j])/10, dtick=int(10*data_full_max[j])/10)


                    if i == 0:
                        fig.update_yaxes(title_text=f"<b>V{j+1}</b>",row=j, col=i+1)
                    if j == num_components - 1:
                        fig.update_xaxes(title_text=f"<b>V{i+1}</b>",row=j, col=i+1)

            update_figure_small(fig)  
        elif num_components == 3:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=pred_traj[:,0], y=pred_traj[:,1], z=pred_traj[:,2], mode='lines', name='<b>prediction</b>', line=dict(width=4, color=cols[0])))
            fig.add_trace(go.Scatter3d(x=target_traj[:,0], y=target_traj[:,1], z=target_traj[:,2], mode='lines', name='<b>ground truth</b>', line=dict(width=3, color=cols[1], dash='dot')))
            fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
                            xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                            yaxis=dict(title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                            zaxis=dict(title='<b>V3</b>', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10)))
            update_figure_3d(fig)
        else:
            fig = go.Figure()
            x_max = np.max(np.abs(target_traj), axis=0)
            fig.add_trace(go.Scatter(x=pred_traj[:,-2], y=pred_traj[:,-1], name='<b>prediction</b>',
                                    mode='lines', line=dict(width=7, color=cols[0])))
            fig.add_trace(go.Scatter(x=target_traj[:,-2], y=target_traj[:,-1], name='<b>ground truth</b>',
                                    mode='lines', line=dict(width=7, color=cols[0], dash='dot')))
            fig.update_layout(xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                        yaxis=dict(scaleanchor="x", scaleratio=1, title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10))
            update_figure(fig)
    
        fig.write_image(os.path.join(save_path, 'trajectories/'+str(traj_id)+'.png'), scale=4)

    def eval_pred(self, data, pred, target, ids, pl_module, extra_steps=0):

        num_components = pl_module.nsv_model.nsv_dim
        save_path = os.path.join(self.task_log_dir, 'mlp_predictions')
        mkdir(save_path)

        full_seq_ids = []
        for i in range(len(ids)):
            if ids[i][1] == extra_steps:
                full_seq_ids.append(i)
        
        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)

        print("Evaluating MLP Predictions")
        mkdir(os.path.join(save_path, 'trajectories'))
        mkdir(os.path.join(save_path, 'time_series'))
        for idx in tqdm(full_seq_ids):
        
            traj_id = ids[idx][0]

            pred_traj = np.concatenate([data[idx:idx+1,:], pred[idx]])
            target_traj = np.concatenate([data[idx:idx+1,:], target[idx]])

            self.plot_trajectory(num_components, target_traj, pred_traj, save_path, traj_id, pl_module.dt, data_max, data_min)

            if pl_module.nsv_model:
                self.visualize(save_path, pred_traj[:,-pl_module.nsv_model.nsv_dim:], target_traj[:,-pl_module.nsv_model.nsv_dim:], traj_id, pl_module.nsv_model)
        
        if pl_module.nsv_model:
            print("Estimating Physical Variables")
            evaluator = Physics_Evaluator(pl_module.nsv_model.dataset)
            evaluator.eval_physics(os.path.join(save_path, 'gt'), ids[full_seq_ids,0], 60, os.path.join(save_path, 'gt/phys_vars.npy'))
            evaluator.eval_physics(os.path.join(save_path, 'pred'), ids[full_seq_ids,0], 60, os.path.join(save_path, 'pred/phys_vars.npy'))
            
            self.compute_physical_error(pl_module.nsv_model.dataset, evaluator.get_phys_vars(), save_path, os.path.join(pl_module.output_dir, pl_module.dataset, 'predictions_long_term', 'base_' + pl_module.nsv_model.seed))
            shutil.rmtree(os.path.join(save_path, 'gt'))
            shutil.rmtree(os.path.join(save_path, 'pred'))
            generate_video_directory(os.path.join(save_path, 'full'), ids[full_seq_ids,0], delete_after=True)


    def compute_physical_error(self, dataset, phys_vars, save_path, baseline_log_dir=None):

        dt = 1/60 if dataset != 'cylindrical_flow' else .02

        if len(phys_vars) == 0:
            return

        base_line_losses = np.load(os.path.join(baseline_log_dir, 'losses.npy'),allow_pickle=True).item()
        baseline_methods = ['hybrid_rollout_3', 'model_rollout']

        gt = np.load(os.path.join(save_path, 'gt/phys_vars.npy'), allow_pickle=True).item()
        phys_variables = np.load(os.path.join(save_path, 'pred/phys_vars.npy'), allow_pickle=True).item()

        losses = {}
        losses['reject'] = phys_variables['reject'].copy()
        losses['reject_data'] = gt['reject'].copy()

        for var in phys_vars:

            losses[var] = []

            for traj_i, traj in enumerate(phys_variables[var]):
                
                traj_loss = []
                
                for i, val in enumerate(traj):
                    if phys_variables['reject'][traj_i][i]:
                        traj_loss.append(np.nan)
                    else:
                        if var in ['theta', 'theta_1', 'theta_2']:
                            diff = calc_theta_diff(gt[var][traj_i][i],val)
                        else:
                            diff = abs(gt[var][traj_i][i] - val)
                        
                        if 'theta' in var:
                            diff = diff * 180 / np.pi
                        
                        traj_loss.append(diff)
                
                losses[var].append(traj_loss)

            losses[var] = np.array(losses[var])
        
        np.save(os.path.join(save_path, 'losses.npy'), losses)
        
        mkdir(os.path.join(save_path, 'plots'))

        fig, axs = plt.subplots(1, len(phys_variables.keys()), layout='constrained', figsize=(8*len(phys_variables.keys()), 6))
        figs = [go.Figure() for _ in range(len(phys_vars) + 1)]

        
        pred_len = phys_variables['reject'].shape[-1]

        if 'reject_data' in phys_variables.keys():
            reject_ratio = scale_reject_ratio(pred_len, phys_variables['reject'], phys_variables['reject_data'])
        else:
            reject_ratio = scale_reject_ratio(pred_len, phys_variables['reject'], np.zeros_like(phys_variables['reject']))

        axs[0].plot(range(pred_len), reject_ratio, label="mlp_integration")
        figs[0].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=reject_ratio, line=dict(color=cols[0], width=4), name='NSVF Integration'))
        for i, method in enumerate(baseline_methods):
            if 'reject_data' in base_line_losses[method].keys():
                base_line_reject_ratio = scale_reject_ratio(pred_len, base_line_losses[method]['reject'], base_line_losses[method]['reject_data'])
            else:
                base_line_reject_ratio = scale_reject_ratio(pred_len, base_line_losses[method]['reject'], np.zeros_like(base_line_losses[method]['reject']))

            axs[0].plot(range(pred_len), base_line_reject_ratio, label='baseline_'+method)
            figs[0].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=base_line_reject_ratio, line=dict(color=cols[i+ 2], width=4), name=f'Non-Smooth {method}'))
    
        
        axs[0].set_title(' Reject Ratio')
        axs[0].set_xlabel("time step")
        axs[0].set_ylabel("Reject Ratio")
        axs[0].set_ylim(-.01,1.05)

        update_figure(figs[0])
        figs[0].update_layout(title='Reject Ratio', 
                            xaxis=dict(title='t', range=[0, dt*(pred_len+1)]),
                            yaxis=dict(title='Reject Ratio', range=[-.01,1.05]), 
                            showlegend=False)

        for i, var in enumerate(phys_vars):
            
            average_loss = np.zeros(pred_len)
            
            for p in range(pred_len):
                error_p = losses[var][:, p]
                error_p = error_p[~np.isnan(error_p)]
                error_p = remove_outlier(error_p)
                if error_p.size > 1:
                    average_loss[p] = np.mean(error_p)
                else:
                    average_loss[p] = np.nan
            
            axs[i+1].plot(range(pred_len), average_loss, label="mlp_integration")
            figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=average_loss, line=dict(color=cols[0], width=4), name=f'NSVF Integration'))


            for j, method in enumerate(baseline_methods):
                average_loss = np.zeros(pred_len)
        
                for p in range(pred_len):
                    error_p = base_line_losses[method][var][:, p]
                    error_p = error_p[~np.isnan(error_p)]
                    error_p = remove_outlier(error_p)
                    if error_p.size > 1:
                        average_loss[p] = np.mean(error_p)
                    else:
                        average_loss[p] = np.nan
                axs[i+1].plot(range(pred_len), average_loss, label=f'baseline_{method}')
                figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=average_loss, line=dict(color=cols[2+j], width=4), name=f'Non-Smooth {method}'))


            axs[i+1].set_title(f'{var} error (L1)')
            axs[i+1].set_xlabel("time step")
            axs[i+1].set_ylabel("L1 error (ignoring nan)")
            
            update_figure(figs[i+1])
            figs[i+1].update_layout(title=f'{var} error (L1)', 
                            xaxis=dict(title='t', range=[0, dt*(pred_len+1)]),
                            yaxis=dict(title="L1 error (ignoring nan)"), 
                            showlegend=False)

        lines,labels = axs[0].get_legend_handles_labels()
        fig.legend(lines,labels , loc='outside lower center', fontsize="20", ncol = len(baseline_methods)+1)
        fig.savefig(os.path.join(save_path, 'plots', f'all.png'))

        phys_vars_list = list(phys_variables.keys())
        for i, f in enumerate(figs):
            f.write_image(os.path.join(save_path, 'plots', f'{phys_vars_list[i]}.png'), scale=4)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name='NSVF Integration', line=dict(color=cols[0])))
        if base_line_losses != None:
            for i, method in enumerate(baseline_methods):
                fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'Non-Smooth {method}', line=dict(color=cols[i + 2])))
        fig.update_layout(xaxis=dict(
                                showline=False,
                                showgrid=False,
                                showticklabels=False,
                                zeroline=False
                            ),
                            yaxis=dict(
                                showline=False,
                                showgrid=False,
                                showticklabels=False,
                                zeroline=False
                            ),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            showlegend=True,
                            legend=dict(traceorder="normal",
                                                font=dict(family="sans serif", size=18, color="black"), x=0.5, y=0.5)
                            )

        # Remove margins
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

        # Save the figure with only the legend
        fig.write_image(os.path.join(save_path, 'plots', 'legend_only.png'))



    def visualize(self, save_path, pred, target, id, nsv_model):


        def get_data(filepath):
            data = Image.open(filepath)
            data = data.resize((128, 128))
            data = np.array(data)
            data = torch.tensor(data / 255.0)
            data = data.permute(2, 0, 1).float()
            return data.cuda()

        if 'smooth' in nsv_model.name:
            nsv_model.eval()
            output, _ = nsv_model.decoder.nsv_decoder(torch.tensor(target).float().cuda())
            nsv_model.eval()
            pred_output, _ = nsv_model.decoder.nsv_decoder(torch.tensor(pred).float().cuda())
        else:
            output, _ = nsv_model.decoder(torch.from_numpy(target).cuda())
            pred_output, _ = nsv_model.decoder(torch.from_numpy(pred).cuda())

        mkpath(os.path.join(save_path, 'full', str(id)))
        mkpath(os.path.join(save_path, 'gt', str(id)))
        mkpath(os.path.join(save_path, 'pred', str(id)))

        #get frames 0,1,2
        suf = os.listdir(f"data/{nsv_model.dataset}/{id}")[0].split('.')[-1]
        frames = [get_data("data/{}/{}/{}.{}".format(nsv_model.dataset, id, i, suf)) for i in range(3)]
        for i in range(2):
            gt = torch.cat([frames[i], frames[i+1]], 2)

            comparison = torch.cat([gt.unsqueeze(0), 
                                    gt.unsqueeze(0),
                                    gt.unsqueeze(0)])

            save_image(comparison.cpu(), os.path.join(save_path,  'full', tuple2name(torch.tensor((id, i)))), nrow=1)

            save_image(frames[i].cpu(), os.path.join(save_path,  'gt', tuple2name(torch.tensor((id, i)))), nrow=1)
            save_image(frames[i].cpu(), os.path.join(save_path,  'pred', tuple2name(torch.tensor((id, i)))), nrow=1)
        
        #get rest of frames
        for idx in range(output.shape[0]):

            gt_1 = get_data("data/{}/{}/{}.{}".format(nsv_model.dataset, id, idx+2, suf))
            gt_2 = get_data("data/{}/{}/{}.{}".format(nsv_model.dataset, id, idx+3, suf))
            gt = torch.cat([gt_1, gt_2], 2)
            comparison = torch.cat([gt.unsqueeze(0), 
                                    output[idx, :, :, :].unsqueeze(0),
                                    pred_output[idx, :, :, ].unsqueeze(0)])
            
            save_image(comparison.cpu(), os.path.join(save_path,  'full', tuple2name(torch.tensor((id, idx+2)))), nrow=1)

            save_image(gt_1.cpu(), os.path.join(save_path,  'gt', tuple2name(torch.tensor((id, idx+2)))), nrow=1)
            save_image(pred_output[idx, :, :, :128].cpu(), os.path.join(save_path,  'pred', tuple2name(torch.tensor((id, idx+2)))), nrow=1)

            if idx == output.shape[0] - 1:
                save_image(gt_2.cpu(), os.path.join(save_path,  'gt', tuple2name(torch.tensor((id, idx+3)))), nrow=1)
                save_image(pred_output[idx, :, :, 128:].cpu(), os.path.join(save_path,  'pred', tuple2name(torch.tensor((id, idx+3)))), nrow=1)

    def visualize_mlp(self, data, pl_module, output, target):

        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)
        data_full_max = np.stack([np.abs(data_min), data_max], axis=0).max(axis=0)

        print("Visualizing NSVF Outputs")
        
        save_path = os.path.join(self.task_log_dir, 'mlp_visualization')
        mkdir(save_path)

        num_components = min(data.shape[1],4)

        for i in tqdm(range(num_components)):

            if num_components == 2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data[:,0], y=data[:,1], mode='markers',
                                        marker=dict(size=8, color=output[:,i], colorbar=dict(title='',tickprefix="<b>",ticksuffix ="</b>", nticks=6),
                                        colorscale=colorscale)))
                fig.update_layout(xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                        yaxis=dict(scaleanchor="x", scaleratio=1, title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                        showlegend=False)
                update_figure(fig,True)       
            else:

                fig = make_subplots(rows=num_components-1, cols=num_components-1)

                for j in range(num_components-1):
                    for k in range(j+1, num_components):
                        fig.add_trace(go.Scatter(x=data[:,j], y=data[:,k], mode='markers',
                                    marker=dict(size=4, color=output[:,i], colorbar=dict(title=''),colorscale=colorscale)), row=k, col=j+1)
                        ig.update_xaxes(row=k, col=j+1, range=[-1.1*data_full_max[j], 1.1*data_full_max[j]], tickmode='linear', tick0 = int(10*data_full_max[j])/10, dtick=int(10*data_full_max[j])/10)
                        fig.update_yaxes(row=k, col=j+1, range=[-1.1*data_full_max[k], 1.1*data_full_max[k]], tickmode='linear', tick0 = int(10*data_full_max[k])/10, dtick=int(10*data_full_max[k])/10)

                        if j == 0:
                            fig.update_yaxes(title_text=f"<b>V{k+1}</b>",row=k, col=j+1)
                        if k == num_components - 1:
                            fig.update_xaxes(title_text=f"<b>V{j+1}</b>",row=k, col=j+1)


                fig.update_layout(showlegend=False)
                update_figure_small(fig)

            fig.write_image(os.path.join(save_path, f'pred_{i+1}.png'), scale=4)
        
        print("Visualizing NSV Finite Differences")

        dt = pl_module.dt
        target_fd = ((target[:, 0] - data) / dt)
        for i in tqdm(range(num_components)):

            if num_components == 2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data[:,0], y=data[:,1], mode='markers',
                                        marker=dict(size=8, color=target_fd[:,i], colorbar=dict(title='',tickprefix="<b>",ticksuffix ="</b>", nticks=6),
                                        colorscale=colorscale)))
                fig.update_layout(xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                        yaxis=dict(scaleanchor="x", scaleratio=1, title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                        showlegend=False)
                update_figure(fig,True)
            else:

                fig = make_subplots(rows=num_components-1, cols=num_components-1)

                for j in range(num_components-1):
                    for k in range(j+1, num_components):
                        fig.add_trace(go.Scatter(x=data[:,j], y=data[:,k], mode='markers',
                                    marker=dict(size=4, color=output[:,i], colorbar=dict(title=''),colorscale=colorscale)), row=k, col=j+1)
                        fig.update_xaxes(row=k, col=j+1, range=[-1.1*data_full_max[j], 1.1*data_full_max[j]], tickmode='linear', tick0 = int(10*data_full_max[j])/10, dtick=int(10*data_full_max[j])/10)
                        fig.update_yaxes(row=k, col=j+1, range=[-1.1*data_full_max[k], 1.1*data_full_max[k]], tickmode='linear', tick0 = int(10*data_full_max[k])/10, dtick=int(10*data_full_max[k])/10)

                        if j == 0:
                            fig.update_yaxes(title_text=f"<b>V{k+1}</b>",row=k, col=j+1)
                        if k == num_components - 1:
                            fig.update_xaxes(title_text=f"<b>V{j+1}</b>",row=k, col=j+1)


                fig.update_layout(showlegend=False)
                update_figure_small(fig)

            fig.write_image(os.path.join(save_path, f'tar_{i+1}.png'), scale=4)
        
        self.visualize_gradField(pl_module, save_path, num_components, data_max, data_min)
        

    def visualize_gradField(self, pl_module, save_path, num_components, data_max, data_min):

        print("Visualizing Gradient Field")

        data_full_max = np.stack([np.abs(data_min), data_max], axis=0).max(axis=0)

        if num_components == 2:

            nsv_range = data_full_max + data_full_max

            g = np.mgrid[0:1:21j,0:1:21j]
            g =  -data_full_max + nsv_range * np.transpose(g, (1,2,0)).reshape(-1,2)

            grid_output = pl_module.model(torch.from_numpy(g).type(torch.FloatTensor).to(device=pl_module.device)).cpu().detach().numpy()

            scale = .03 if pl_module.dataset == 'single_pendulum' else .02
            fig = ff.create_quiver(g[:,0], g[:,1], grid_output[:,0], grid_output[:,1], scale=scale, line=dict(color='black'), line_width=3)

            fig.update_layout(width=700, height=660,
                                xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                                yaxis=dict(title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                                showlegend=False)
            
            update_figure(fig,True)
            fig.write_image(os.path.join(save_path, 'gradient_field.png'), scale=4)
        
        elif num_components == 4:

            nsv_range = data_full_max + data_full_max

            eq_path = os.path.join(self.task_log_dir, 'mlp_equilibrium', 'eq_points.npy')
            equilibriums = np.load(eq_path, allow_pickle=True).item()

            num_eqs = len(equilibriums["roots"])

            equilibrium = None
            for i in range(num_eqs):
                if equilibriums['stabilities'][i] == "stable" and equilibriums['successes'][i] == True:
                    equilibrium = equilibriums['roots'][i]
                    break
            if not isinstance(equilibrium, np.ndarray):
                equilibrium = np.zeros(4)
                    
            print(equilibrium)

            g = np.mgrid[0:1:21j,0:1:21j]
            g = -data_full_max[:2] + nsv_range[:2] * np.transpose(g, (1,2,0)).reshape(-1,2)
            

            equilibrium_v1v2 = np.broadcast_to(equilibrium[2:], g.shape)

            print(equilibrium_v1v2.shape)

            g_v1v2 = np.concatenate([g,equilibrium_v1v2], axis=1)
            print(g_v1v2.shape)

            grid_output_v1v2 = pl_module.model(torch.from_numpy(g_v1v2).type(torch.FloatTensor).to(device=pl_module.device)).cpu().detach().numpy()

            fig = ff.create_quiver(g[:,0], g[:,1], grid_output_v1v2[:,0], grid_output_v1v2[:,1], scale=.03, line=dict(color='black'), line_width=3)

            fig.update_layout(width=700, height=660,
                                xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = -int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                                yaxis=dict(title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = -int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                                showlegend=False)
            update_figure(fig, True)
            fig.write_image(os.path.join(save_path, 'gradient_field_v1v2.png'), scale=4)

            g = np.mgrid[0:1:21j,0:1:21j]
            g = -data_full_max[2:] + nsv_range[2:] * np.transpose(g, (1,2,0)).reshape(-1,2)

            equilibrium_v3v4 = np.broadcast_to(equilibrium[:2], g.shape)

            g_v3v4 = np.concatenate([equilibrium_v3v4, g], axis=1)

            grid_output_v3v4 = pl_module.model(torch.from_numpy(g_v3v4).type(torch.FloatTensor).to(device=pl_module.device)).cpu().detach().numpy()

            fig = ff.create_quiver(g[:,0], g[:,1], grid_output_v3v4[:,2], grid_output_v1v2[:,3], scale=.03, line=dict(color='black'), line_width=3)

            update_figure(fig, True)
            fig.update_layout(width=700, height=660,
                                xaxis=dict(title='<b>V3</b>', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10),
                                yaxis=dict(title='<b>V4</b>', range=[-1.1*data_full_max[3], 1.1*data_full_max[3]], tickmode='linear', tick0 = int(10*data_full_max[3])/10, dtick=int(10*data_full_max[3])/10),
                                showlegend=False)
            fig.write_image(os.path.join(save_path, 'gradient_field_v3v4.png'), scale=4)
        
        else:

            nsv_range = data_full_max + data_full_max
            
            eq_path = os.path.join(self.task_log_dir, 'mlp_equilibrium', 'eq_points.npy')
            equilibriums = np.load(eq_path, allow_pickle=True).item()

            num_eqs = len(equilibriums["roots"])

            equilibrium = None
            for i in range(num_eqs):
                if equilibriums['stabilities'][i] == "stable" and equilibriums['successes'][i] == True:
                    equilibrium = equilibriums['roots'][i]
                    break
            if not isinstance(equilibrium, np.ndarray):
                equilibrium = np.zeros(3)
                    
            g = np.mgrid[0:1:21j,0:1:21j]
            g = -data_full_max[:2] + nsv_range[:2] * np.transpose(g, (1,2,0)).reshape(-1,2)

            equilibrium_v1v2 = np.broadcast_to(equilibrium[2], (g.shape[0],1))

            g_v1v2 = np.concatenate([g,equilibrium_v1v2], axis=1)

            grid_output_v1v2 = pl_module.model(torch.from_numpy(g_v1v2).type(torch.FloatTensor).to(device=pl_module.device)).cpu().detach().numpy()

            fig = ff.create_quiver(g[:,0], g[:,1], grid_output_v1v2[:,0], grid_output_v1v2[:,1], scale=.1, line=dict(color='black'), line_width=3)

            update_figure(fig, True)
            fig.update_layout(width=700, height=660,
                                xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                                yaxis=dict( title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                                showlegend=False)

            fig.write_image(os.path.join(save_path, 'gradient_field_v1v2.png'), scale=4)


            g = np.mgrid[0:1:21j,0:1:21j]
            g = -data_full_max[1:] + nsv_range[1:] * np.transpose(g, (1,2,0)).reshape(-1,2)
            equilibrium_v2v3 = np.broadcast_to(equilibrium[0], (g.shape[0],1))

            g_v2v3 = np.concatenate([equilibrium_v2v3, g], axis=1)

            grid_output_v2v3 = pl_module.model(torch.from_numpy(g_v2v3).type(torch.FloatTensor).to(device=pl_module.device)).cpu().detach().numpy()

            fig = ff.create_quiver(g[:,0], g[:,1], grid_output_v2v3[:,1], grid_output_v2v3[:,2], scale=.1, line=dict(color='black'), line_width=3)

            update_figure(fig,True)
            fig.update_layout(width=700, height=660,
                                xaxis=dict(title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                                yaxis=dict( title='<b>V3</b>', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10),
                                showlegend=False)

            fig.write_image(os.path.join(save_path, 'gradient_field_v2v3.png'), scale=4)
    

            g = np.mgrid[0:1:21j,0:1:21j]
            g = np.array([-data_full_max[0],-data_full_max[2]]) + np.array([nsv_range[0],nsv_range[2]])* np.transpose(g, (1,2,0)).reshape(-1,2)
            equilibrium_v1v3 = np.broadcast_to(equilibrium[1], (g.shape[0],1))

            g_v1v3 = np.concatenate([g[:,0:1], equilibrium_v1v3, g[:,-1:]], axis=1)

            grid_output_v1v3 = pl_module.model(torch.from_numpy(g_v1v3).type(torch.FloatTensor).to(device=pl_module.device)).cpu().detach().numpy()

            fig = ff.create_quiver(g[:,0], g[:,1], grid_output_v1v3[:,0], grid_output_v1v3[:,2], scale=.1, line=dict(color='black'), line_width=3)

            update_figure(fig, True)
            fig.update_layout(width=700, height=660,
                                xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                                yaxis=dict( title='<b>V3</b>', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10),
                                showlegend=False)
            fig.write_image(os.path.join(save_path, 'gradient_field_v1v3.png'), scale=4)

    def trajectories_from_data_ids(self, ids, data, output):
        id2index = {tuple(id): i for i, id in enumerate(ids)}
        data_trajectories = collections.defaultdict(list)
        output_trajectories = collections.defaultdict(list)

        for id in sorted(id2index.keys()):
            i = id2index[id]
            data_trajectories[id[0]].append(data[i])
            output_trajectories[id[0]].append(output[i])

        return data_trajectories, output_trajectories


    def find_equilibrium_mlp(self, ids, data, pl_module, output, num_guesses=10, num_samples=10, num_steps=60, epsilon_percentages=[], delta_percentages=[]):

        save_path = os.path.join(self.task_log_dir, f'mlp_equilibrium')
        mkdir(save_path)

        def mlp_f(y): return pl_module.model(torch.tensor(y.astype(np.float32)).unsqueeze(0).cuda()).cpu().detach().numpy().squeeze()

        indices = np.argsort(np.linalg.norm(output, axis=1))
        guesses = data[indices[:num_guesses]]
        
        # Candidate per trajectory
        if pl_module.dataset == 'cylindrical_flow':

            data_trajectories, output_trajectories = self.trajectories_from_data_ids(ids, data, output)

            guesses = []
            traj_nums = []

            for traj in data_trajectories.keys():

                guesses.append(np.array(data_trajectories[traj][-10:]).mean(axis=0))
                traj_nums.append(traj)

        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)
        nsv_range = data.max() - data.min()

        roots = []
        eigenvals = []
        jacobians = []
        distances = []
        initial_distances = []
        validity = []
        successes = []

        print("Finding Equilibrium Points")

        for guess_idx, point in tqdm(enumerate(guesses)):

            mlp_root, _, ier, mesg = fsolve(mlp_f, point, full_output=True, xtol=1e-6)
            if ier == 1:
                success = True
            else:
                success = False

            if np.all(mlp_root < data_max) and np.all(mlp_root > data_min):
                valid = True
            else:
                valid = False

            mlp_nsv = torch.tensor(mlp_root.astype(np.float32)).reshape((1,-1)).cuda()
            
            if "smooth" in pl_module.nsv_model.name:
                image, _, _ = pl_module.nsv_model.decoder(mlp_nsv)
            else:
                image, _ = pl_module.nsv_model.decoder(mlp_nsv)
            

            with torch.enable_grad():
                mlp_nsv.requires_grad = True
                output = pl_module.model(mlp_nsv)
                jac = computeJacobian(output, mlp_nsv).squeeze().cpu().detach().numpy()

                ev = eigvals(jac)
                ev_real = np.array([complex(e).real for e in ev])

            distance = []
            initial_distance = []

            for p in delta_percentages:
                delta = .01 * p * nsv_range
                distance.append([])
                initial_distance.append([])
                for i in range(num_samples):
                    max_dist, initial_dist = self.mlp_pred_equilibrium_sample(i, pl_module, guess_idx, torch.tensor(mlp_root.astype(np.float32)).reshape((1,-1)).cuda(), steps=num_steps, delta=delta)
                    distance[-1].append(max_dist)
                    initial_distance[-1].append(initial_dist)

            mkdir(os.path.join(save_path, f'{guess_idx}'))
            save_image(image.cpu(), os.path.join(save_path, f'{guess_idx}.png'), nrow=1)
            save_image(image[:,:,:,:128].cpu(), os.path.join(save_path, f'{guess_idx}/0.png'), nrow=1)
            save_image(image[:,:,:,128:].cpu(), os.path.join(save_path, f'{guess_idx}/1.png'), nrow=1)

            roots.append(mlp_root)
            eigenvals.append(ev)
            jacobians.append(jac)
            distances.append(distance)
            initial_distances.append(initial_distance)
            validity.append(valid)
            successes.append(success)

        print("Annotating Equilibrium Images")
        evaluator = Physics_Evaluator(pl_module.nsv_model.dataset)
        marked_images = evaluator.eval_physics(save_path, range(len(guesses)), 2, os.path.join(save_path, 'phys_vars.npy'), return_marked=True)

        if marked_images != None:
            for guess_idx, m_i in tqdm(enumerate(marked_images)):
                cv2.imwrite(os.path.join(save_path, f'{guess_idx}/m_0.png'), m_i[0])
                cv2.imwrite(os.path.join(save_path, f'{guess_idx}/m_1.png'), m_i[1])
        else:
            for guess_idx, m_i in tqdm(enumerate(guesses)):
                os.system(f'cp \"{save_path}/{guess_idx}/0.png\" \"{save_path}/{guess_idx}/m_0.png\"')
                os.system(f'cp \"{save_path}/{guess_idx}/1.png\" \"{save_path}/{guess_idx}/m_1.png\"')
        

        delta_per_epsilon, stabilities = self.eq_stability_analysis(initial_distances, distances, nsv_range, epsilon_percentages)
        eq_points = {"guesses": guesses,
                            "roots": roots,
                            "stabilities": stabilities,
                            "eigenValues": eigenvals,
                            "jacobians": jacobians,
                            "distances": distances,
                            "initial_distances": initial_distances,
                            "validity": validity,
                            "successes": successes,
                            "delta_per_epsilon": delta_per_epsilon
                            }
        eq_points = np.array(eq_points)
        np.save(os.path.join(save_path, 'eq_points.npy'), eq_points)
            
    
    def eq_stability_analysis(self, initial_distances, distances, nsv_range, epsilon_percentages):

        delta_per_epsilon = []
        stabilities = []

        print("Analyzing Stabilities")
        for i_d, max_d in tqdm(zip(initial_distances, distances)):

            initial_d = np.array(i_d).flatten()
            maximum_d = np.array(max_d).flatten()

            i_d_2_max_d = {i: d for i, d in zip(initial_d, maximum_d)}
            sorted_i_d_2_max_d = sorted(i_d_2_max_d, key = i_d_2_max_d.get)

            result = []

            for p in epsilon_percentages:
                e = .01 * p * nsv_range
                
                result_p = None

                for i in sorted_i_d_2_max_d:

                    d = i_d_2_max_d[i]
                    d_percentage = "{:.2f}".format(100 * d/nsv_range)

                    if d  < e:
                        result_p = f'epsilon_{p}%_{e}_delta_{d_percentage}%_{d}'
                    else:
                        break
                
                result.append(result_p)
            
            delta_per_epsilon.append(result)
            stabilities.append("unstable" if None in result else "stable")
        
        return delta_per_epsilon, stabilities
    
    def mlp_pred_equilibrium_sample(self, trial_num, pl_module, root_idx, begin_nsv, steps=60, delta=0.01):

        nsv_sample = torch.normal(mean=begin_nsv, std=delta)

        initial_dist = torch.sqrt(torch.sum(((nsv_sample - begin_nsv)**2)))

        while initial_dist >= delta:

            nsv_sample = torch.normal(mean=begin_nsv, std=delta)

            initial_dist = torch.sqrt(torch.sum(((nsv_sample - begin_nsv)**2)))

        t_span = (pl_module.dt * torch.arange(steps+1)).float()
        
        _, pred = pl_module.ode(nsv_sample.cuda(), t_span)

        pred = pred.permute(1, 0, 2).squeeze()

        pred_arr = pred.clone().detach().cpu().numpy()
        
        pred_distance = torch.sqrt(torch.sum((pred - begin_nsv)**2, dim=1))

        return pred_distance.max().cpu().detach().numpy(), initial_dist.cpu().detach().numpy()

    def rollout_equilibrium(self, pl_module, root_idx, begin_data, steps=60, hybrid_step=3):

        root = f'{root_idx}'
        save_path = os.path.join(self.task_log_dir, 'mlp_equilibrium')
        mkdir(os.path.join(save_path, root + '_rollout'))
        
        data = begin_data

        for i in range(steps):

            if (i+1)%hybrid_step == 0:
                if "smooth" in pl_module.nsv_model.name:
                    output, latent, state_reconstructured, state, state_gt, latent_gt = pl_module.nsv_model(data)
                else:
                    output, latent, state, latent_gt = pl_module.nsv_model(data)
            else:

                if "smooth" in pl_module.nsv_model.name:
                    output = pl_module.nsv_model.decoder.nsv_decoder.latent_decoder(pl_module.nsv_model.encoder.nsv_encoder.latent_encoder(data))
                else:
                    output = pl_module.nsv_model.decoder.latent_decoder(pl_module.nsv_model.encoder.latent_encoder(data))
            
            save_image(output[:,:,:,:128].cpu(), os.path.join(save_path, f'{root}_rollout/{2*i}.png'))
            save_image(output[:,:,:,128:].cpu(), os.path.join(save_path, f'{root}_rollout/{2*i+1}.png'))

            data = output
        
        generate_video(save_path, root + '_rollout', os.path.join(save_path, f'{root}_rollout.mp4'), delete_after=True)


    def on_test_end(self, trainer, pl_module):
        data = np.array(pl_module.all_data)
        output = np.array(pl_module.all_outputs)
        ids = np.array(pl_module.all_filepaths)
        pred = np.array(pl_module.all_preds)
        target = np.array(pl_module.all_targets)

        
        np.save(os.path.join(pl_module.var_log_dir, 'ids.npy'), pl_module.all_filepaths)
        np.save(os.path.join(pl_module.var_log_dir, 'data.npy'), data)
        np.save(os.path.join(pl_module.var_log_dir, 'output.npy'), output)

        if pl_module.test_mode == "save_train_data":
            return

        output_norm = np.max(np.abs(output), axis=0)
        print('Gradient max: ', output_norm)

        self.task_log_dir = os.path.join(pl_module.output_dir, pl_module.dataset, 'tasks', pl_module.model_name)

        self.find_equilibrium_mlp(ids, data, pl_module, output,  num_steps=60, epsilon_percentages=sorted(pl_module.percentages, reverse=True), delta_percentages=sorted(pl_module.percentages))

        if pl_module.test_mode == "eq_stability":
            return
        
        self.eval_pred(data, pred, target, ids, pl_module, extra_steps=pl_module.extra_steps)
        self.visualize_mlp(data, pl_module, output, target)