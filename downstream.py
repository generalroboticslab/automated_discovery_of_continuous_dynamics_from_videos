from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import os
import pprint
import argparse
import numpy as np
from munch import munchify
import random

from models.vis_dynamics_model import *
from models.nsv_dynamics_model import *
from models.nsv_mlp import *
from models.data_module import *
from models.callbacks import *

from utils.misc import *
from utils.analysis import Physics_Evaluator

from pytorch_lightning import Trainer
from torchdyn.core import NeuralODE
from torchvision.utils import save_image


def trajectories_from_data_ids(ids, data, output):
    id2index = {tuple(id): i for i, id in enumerate(ids)}
    data_trajectories = collections.defaultdict(list)
    output_trajectories = collections.defaultdict(list)

    for id in sorted(id2index.keys()):
        i = id2index[id]
        data_trajectories[id[0]].append(data[i])
        output_trajectories[id[0]].append(output[i])
    
    for id in data_trajectories.keys():
        data_trajectories[id] =  np.array(data_trajectories[id])
        output_trajectories[id] = np.array(output_trajectories[id])

    return data_trajectories, output_trajectories

def data_trajectories_from_data_ids(ids, data):
    id2index = {tuple(id): i for i, id in enumerate(ids)}
    data_trajectories = collections.defaultdict(list)

    for id in sorted(id2index.keys()):
        i = id2index[id]
        data_trajectories[id[0]].append(data[i])
    
    for id in data_trajectories.keys():
        data_trajectories[id] =  np.array(data_trajectories[id])

    return data_trajectories


def plot_trajectory(target_traj, pred_traj, save_path, traj_id, pred_name='prediction', target_name='ground_truth'):

    num_components = pred_traj.shape[1]

    if num_components != 2:
        fig = make_subplots(rows=num_components-1, cols=num_components-1)
        x_max = np.max(np.abs(target_traj), axis=0)
        for i in range(num_components-1):
            for j in range(i+1, num_components):
                fig.add_trace(go.Scatter(x=pred_traj[:,i], y=pred_traj[:,j], name=pred_name,
                                mode='lines', line=dict(width=1, color=cols[0])), row=j, col=i+1)
                fig.add_trace(go.Scatter(x=target_traj[:,i], y=target_traj[:,j], name=target_name,
                            mode='lines', line=dict(width=0.75, color=cols[1])), row=j, col=i+1)
                fig.update_xaxes(row=j, col=i+1, range=[-1.3 * x_max[i], 1.3 * x_max[i]])
                fig.update_yaxes(row=j, col=i+1, range=[-1.3 * x_max[j], 1.3 * x_max[j]])    

                if i == 0:
                    fig.update_yaxes(title_text=f"x{j}",row=j, col=i+1)
                if j == num_components - 1:
                    fig.update_xaxes(title_text=f"x{i}",row=j, col=i+1)

        update_figure_small(fig)  
    else:
        fig = go.Figure()
        x_max = np.max(np.abs(target_traj), axis=0)
        fig.add_trace(go.Scatter(x=pred_traj[:,0], y=pred_traj[:,1], name=pred_name,
                                mode='lines', line=dict(width=4, color=cols[0])))
        fig.add_trace(go.Scatter(x=target_traj[:,0], y=target_traj[:,1], name=target_name,
                                mode='lines', line=dict(width=3, color=cols[1])))
        fig.update_xaxes(range=[-1.3 * x_max[0], 1.3 * x_max[0]])
        fig.update_yaxes(range=[-1.3 * x_max[1], 1.3 * x_max[1]])
    
        update_figure(fig)

    fig.update_layout(title=f'video no. {traj_id}')
    fig.write_image(os.path.join(save_path, 'trajectories/plot_pred_'+str(traj_id)+'.png'), scale=4)


def prepare_Model(args, damping=0.0, reverse=False):

    seed_everything(args.seed)

    if 'regressDeeper' in args.model_name:
        model = DeeperNSVMLP(nsv_dim=get_experiment_dim(args.dataset, args.seed), **args)
    else:
        model = NSVMLP(nsv_dim=get_experiment_dim(args.dataset, args.seed), **args)

    if 'phys' not in args.nsv_model_name:
        if 'smooth' in args.nsv_model_name:
            nsv_model = SmoothNSVAutoencoder.from_model_name(args.nsv_model_name, **args)
        else:
            nsv_model = NSVAutoencoder.from_model_name(args.nsv_model_name, **args)
    else:
        nsv_model = None

    
    weight_path = get_weightPath(args, False)
    print("Testing: ", weight_path)

    if weight_path == None:
        exit("No Model Saved")
    
    args.model_name = create_name(args)
    net = NSVDynamicsModel.load_from_checkpoint(weight_path, model=model, nsv_model=nsv_model,  **args)

    regress_path = os.path.join(args.output_dir, args.dataset, 'tasks', args.model_name)
    eq_path = os.path.join(regress_path, 'mlp_equilibrium', 'eq_points.npy')

    eqs = np.load(eq_path, allow_pickle=True).item()

    equilibrium = None
    for i, s in enumerate(eqs['stabilities']):
        if s == "stable" and eqs['validity'][i] == True and eqs['successes'][i] == True:
            equilibrium = eqs['roots'][i]
            break

    if not type(equilibrium) is np.ndarray:
        print("Redo search")
        minDistance = 100000
        for i, d in enumerate(eqs['distances']):
            d_arr = np.array(d)
            if d_arr.max() < minDistance and eqs['validity'][i] == True and eqs['successes'][i] == True:
                equilibrium = eqs['roots'][i]
                minDistance = d_arr.max()
                
        print(equilibrium)

    assert type(equilibrium) is np.ndarray
    damped_model = DampedNSVMLP(net.model, equilibrium, damping=damping, reverse=reverse)
    
    
    return net.nsv_model, damped_model, net.model

def concatenate_images_horizontally(image_paths, output_path):
    images = [Image.open(path) for path in image_paths]
    
    # Find the maximum height among all images
    max_height = max(img.size[1] for img in images)
    
    # Calculate total width
    total_width = sum(img.size[0] for img in images)
    
    # Create a new image with the total width and max height
    new_image = Image.new('RGB', (total_width, max_height))
    
    # Paste images horizontally
    current_width = 0
    for img in images:
        # If image height is less than max_height, paste it centered vertically
        if img.size[1] < max_height:
            y_offset = (max_height - img.size[1]) // 2
            new_image.paste(img, (current_width, y_offset))
        else:
            new_image.paste(img, (current_width, 0))
        current_width += img.size[0]
    
    # Save the result
    new_image.save(output_path)
    print(f"Concatenated image saved as {output_path}")

def plot_with_gradField(trajectories, save_path, model, dt, output_filename, equilibrium, data_min, data_max):

    num_components = trajectories[0].shape[-1]

    traj_arr = np.array(trajectories).reshape(-1,num_components)
    x_max = np.max(np.abs(traj_arr-equilibrium), axis=0)

    if num_components != 2:
        fig = make_subplots(rows=num_components-1, cols=num_components-1)

        x_range = 2 * x_max
        x_min_13 = equilibrium - 1.3 * x_max
        
        for i, traj in enumerate(trajectories):
            for j in range(num_components-1):
                for k in range(j+1, num_components):
                    fig.add_trace(go.Scatter(x=traj[:,j], y=traj[:,k], showlegend=False,
                                    mode='lines', line=dict(width=3, color=cols[i])), row=k, col=j+1)
                    fig.update_xaxes(row=k, col=j+1, range=[equilibrium[j] - 1.3 * x_max[j], equilibrium[j] + 1.3 * x_max[j]], tickmode='linear', tick0 = int(10*(equilibrium[j]-x_max[j]))/10, dtick = int(10*x_max[j])/10, showticklabels=(k==num_components - 1))
                    fig.update_yaxes(row=k, col=j+1, range=[equilibrium[k] - 1.3 * x_max[k], equilibrium[k] + 1.3 * x_max[k]], tickmode='linear', tick0 = int(10*(equilibrium[k]-x_max[k]))/10, dtick = int(10*x_max[k])/10, showticklabels=(j==0))    

                    # if j == 0:
                    #     fig.update_yaxes(title_text=f"<b>V{k+1}</b>",row=k, col=j+1)
                    # if k == num_components - 1:
                    #     fig.update_xaxes(title_text=f"<b>V{j+1}</b>",row=k, col=j+1)

        update_figure_small(fig, True)  
    else:

        x_max_13 = np.min(np.stack([equilibrium + 1.3 * x_max, data_max], axis=0),axis=0)
        x_min_13 = np.max(np.stack([equilibrium - 1.3 * x_max, data_min], axis=0),axis=0)
        x_range = x_max_13 - x_min_13

        g = np.mgrid[0:1:15j,0:1:15j]
        g = x_min_13 + x_range * np.transpose(g, (1,2,0)).reshape(-1,2)

        grid_output = model(torch.from_numpy(g).type(torch.FloatTensor).cuda()).cpu().detach().numpy()

        fig = ff.create_quiver(g[:,0], g[:,1], grid_output[:,0], grid_output[:,1], scale=.02, line=dict(color='grey'))
        # fig.update_layout(xaxis=dict(title='<b>V1</b>', range=[equilibrium[0] - 1.3 * x_max[0], equilibrium[0] + 1.3 * x_max[0]], tickmode='linear', tick0 = int(10*(equilibrium[0]-x_max[0]))/10, dtick = int(10*x_max[0])/10),
        #                         yaxis=dict(title='<b>V2</b>', range=[equilibrium[1] - 1.3 * x_max[1], equilibrium[1] + 1.3 * x_max[1]], tickmode='linear', tick0 = int(10*(equilibrium[1]-x_max[1]))/10, dtick = int(10*x_max[1])/10), 
        #                         width=700, height=640, showlegend=False)
        fig.update_layout(xaxis=dict(range=[x_min_13[0], x_min_13[0] + x_range[0]], tickmode='linear', tick0 = int(10*(x_min_13[0]))/10, dtick = int(10*x_range[0]/3)/10),
                                yaxis=dict(range=[x_min_13[1], x_min_13[1] + x_range[1]], tickmode='linear', tick0 = int(10*(x_min_13[1]))/10, dtick = int(10*x_range[1]/3)/10), 
                                width=700, height=640, showlegend=False)

        steps = trajectories[0].shape[0]

        fig_v = make_subplots(rows=2, cols=1)
        fig_v.update_xaxes(range=[0,dt*(steps+1)], nticks=3, row=1)
        fig_v.update_yaxes(range=[x_min_13[0], x_min_13[0] + x_range[0]], tickmode='linear', tick0 = int(10*(x_min_13[0]))/10, dtick = int(10*x_range[0]/3)/10, row=1, col=1)
        fig_v.update_xaxes(range=[0,dt*(steps+1)], nticks=3, row=2)
        fig_v.update_yaxes(range=[x_min_13[1], x_min_13[1] + x_range[1]], tickmode='linear', tick0 = int(10*(x_min_13[1]))/10, dtick = int(10*x_range[1]/3)/10, row=2,col=1)
        

        # fig_v = make_subplots(rows=2, cols=1)
        # fig_v1.update_layout(xaxis=dict(title='<b>Time (s)</b>', range=[0,dt*(steps+1)],  nticks=3),
        #                         yaxis=dict(title='<b>V1</b>', range=[equilibrium[0] - 1.3 * x_max[0], equilibrium[0] + 1.3 * x_max[0]], tickmode='linear', tick0 = int(10*(equilibrium[0]-x_max[0]))/10, dtick = int(10*x_max[0])/10), showlegend=False)
        
        # fig_v2.update_layout(xaxis=dict(title='<b>Time (s)<b>', range=[0,dt*(steps+1)],  nticks=3),
        #                         yaxis=dict(title='<b>V2</b>', range=[equilibrium[1] - 1.3 * x_max[1], equilibrium[1] + 1.3 * x_max[1]], tickmode='linear', tick0 = int(10*(equilibrium[1]-x_max[1]))/10, dtick = int(10*x_max[1])/10), showlegend=False)
        
        for i, traj in enumerate(trajectories):
            fig.add_trace(go.Scatter(x=traj[:,0], y=traj[:,1], 
                                    mode='lines', line=dict(width=7, color=cols[i])))

            fig_v.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=traj[:,0], line=dict(color=cols[i], width=7), showlegend=False), row=1,col=1)
            fig_v.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=traj[:,1], line=dict(color=cols[i], width=7), showlegend=False), row=2,col=1)


        fig_v.update_layout(showlegend=False, width=700, height=700)
        update_figure(fig, True)
        update_figure(fig_v, True)
        fig_v.write_image(os.path.join(save_path, f'V_{output_filename}'), scale=4)

    fig.write_image(os.path.join(save_path, output_filename), scale=4)

def make_rand_vector(dims):
    vec = [random.gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def damping(args, damping=0.5, num_videos=6, video_len=2, steps=120, delta_percentage=50):

    nsv_model, damped_model, undamped_model = prepare_Model(args, damping=damping)
    damped_ode = NeuralODE(damped_model, solver='rk4').cuda()
    undamped_ode = NeuralODE(undamped_model, solver='rk4').cuda()
    nsv_model.cuda()

    data_path = os.path.join(args.output_dir, args.dataset, 'variables', args.model_name, 'data.npy')
    data = np.load(data_path)

    data_max = np.max(data, axis=0)
    data_max_torch = torch.tensor(data_max).float().cuda().reshape(1,-1)
    data_min = np.min(data, axis=0)
    data_min_torch = torch.tensor(data_min).float().cuda().reshape(1,-1)
    nsv_range = np.sqrt(np.sum((data_max - data_min)**2))

    print(damped_model.equilibrium)

    save_path = os.path.join(args.output_dir, args.dataset, "downstream", args.model_name, f'damping', f'damping_{damping}_videoLen_{video_len}_steps_{steps}')
    mkdir(save_path) #PATH

    t_span = (video_len / steps * torch.arange(steps)).float()

    delta = .01 * delta_percentage * nsv_range
    perturbations = delta / num_videos

    seed_everything(args.seed)
    mkdir(os.path.join(save_path, 'trajectories'))
    mkdir(os.path.join(save_path, 'all_damped'))
    mkdir(os.path.join(save_path, 'all_undamped'))
    success = False

    start_values = []
    undamped_trajectories = []
    damped_trajectories = []
    while success != True:

        rand_dir = torch.tensor(make_rand_vector(nsv_model.nsv_dim)).float().cuda()

        start_values_trial = []
        undamped_trajectories_trial = []
        damped_trajectories_trial = []

        for i in tqdm(range(num_videos)):
            
            nsv_sample = damped_model.equilibrium + rand_dir * (i + 1) * perturbations
            nsv_sample = nsv_sample.unsqueeze(0)

            _, damped_pred = damped_ode(nsv_sample, t_span)
            _, undamped_pred = undamped_ode(nsv_sample, t_span)

            if torch.any(damped_pred > data_max_torch) or torch.any(damped_pred < data_min_torch) or torch.any(undamped_pred > data_max_torch) or torch.any(undamped_pred < data_min_torch):
                print("Escaped Boundary, Try again")
                break

            damped_pred = damped_pred.permute(1, 0, 2).squeeze()
            undamped_pred = undamped_pred.permute(1, 0, 2).squeeze()

            start_values_trial.append(nsv_sample)
            damped_trajectories_trial.append(damped_pred)
            undamped_trajectories_trial.append(undamped_pred)

            if i == num_videos - 1:
                success = True

        if success:
            for i in tqdm(range(num_videos)):
                damped_pred = damped_trajectories_trial[i]
                undamped_pred = undamped_trajectories_trial[i]
                start_value = start_values_trial[i]

                start_values.append(start_value.cpu().detach().numpy())
                undamped_trajectories.append(undamped_pred.cpu().detach().numpy())
                damped_trajectories.append(damped_pred.cpu().detach().numpy())

                plot_trajectory(undamped_pred.cpu().detach().numpy(), damped_pred.cpu().detach().numpy(), save_path, i, pred_name='damped', target_name='undamped')
                
                if "smooth" in nsv_model.name:
                    damped_pred_output, _, _ = nsv_model.decoder(damped_pred)
                    undamped_pred_output, _, _ = nsv_model.decoder(undamped_pred)
                else:
                    damped_pred_output, _ = nsv_model.decoder(damped_pred)
                    undamped_pred_output, _ = nsv_model.decoder(undamped_pred)
                
                mkdir(os.path.join(save_path, f'damped_{i}'))
                mkdir(os.path.join(save_path, f'undamped_{i}'))
                for idx in range(damped_pred_output.shape[0]):

                    save_image(damped_pred_output[idx, :, :, :128].cpu(), os.path.join(save_path,  f'damped_{i}/{idx}.png'), nrow=1)
                    save_image(undamped_pred_output[idx, :, :, :128].cpu(), os.path.join(save_path,  f'undamped_{i}/{idx}.png'), nrow=1)
                
                generate_video(save_path, f'damped_{i}', os.path.join(save_path, f'damped_{i}.mp4'))
                generate_video(save_path, f'undamped_{i}', os.path.join(save_path, f'undamped_{i}.mp4'))
                
                os.system(f'cp -r {save_path}/damped_{i} {save_path}/all_damped/{i}')
                os.system(f'cp -r {save_path}/undamped_{i} {save_path}/all_undamped/{i}')

    evaluator = Physics_Evaluator(args.dataset)
    marked_images_damped = evaluator.eval_physics(os.path.join(save_path, f'all_damped'), range(num_videos), steps, os.path.join(save_path, f'damped.npy'), return_marked=True)
    for vid_id, vid in enumerate(marked_images_damped):
        mkdir(os.path.join(save_path, f'damped_{vid_id}_marked'))
        for idx in range(len(vid)):
            cv2.imwrite(os.path.join(save_path, f'damped_{vid_id}_marked/{idx}_m.png'), vid[idx])

    marked_images_undamped = evaluator.eval_physics(os.path.join(save_path, f'all_undamped'), range(num_videos), steps, os.path.join(save_path, f'undamped.npy'), return_marked=True)
    for vid_id, vid in enumerate(marked_images_undamped):
        mkdir(os.path.join(save_path, f'undamped_{vid_id}_marked'))
        for idx in range(len(vid)):
            cv2.imwrite(os.path.join(save_path, f'undamped_{vid_id}_marked/{idx}_m.png'), vid[idx])
    
    damped_trajectories = sort_trajectories(damped_trajectories, damped_model.equilibrium.clone().detach().cpu().numpy())
    plot_with_gradField(damped_trajectories, save_path, damped_model, video_len / steps, 'damped_trajectories.png', damped_model.equilibrium.clone().detach().cpu().numpy(), data_min, data_max)
    undamped_trajectories = sort_trajectories(undamped_trajectories, damped_model.equilibrium.clone().detach().cpu().numpy())
    plot_with_gradField(undamped_trajectories, save_path, undamped_model, video_len / steps, 'undamped_trajectories.png', damped_model.equilibrium.clone().detach().cpu().numpy(), data_min, data_max)


    # Get all image files from the directory
    image_files = [str(p)+'_m.png' for p in range(0, steps, 10)]
    
    for vid_id in range(num_videos):
        # Create full paths for images
        damped_image_paths = [os.path.join(save_path, f'damped_{vid_id}_marked', img) for img in image_files]
        undamped_image_paths =  [os.path.join(save_path, f'undamped_{vid_id}_marked', img) for img in image_files]

        damped_output_path = os.path.join(save_path, f'damped_{vid_id}_marked.png')
        undamped_output_path = os.path.join(save_path, f'undamped_{vid_id}_marked.png')

        concatenate_images_horizontally(damped_image_paths, damped_output_path)
        concatenate_images_horizontally(undamped_image_paths, undamped_output_path)

    start_values = np.array(start_values)
    damped_trajectories = np.array(damped_trajectories)
    undamped_trajectories = np.array(undamped_trajectories)
    np.save(os.path.join(os.path.join(save_path, 'start_values.npy')), start_values)
    np.save(os.path.join(os.path.join(save_path, 'damped_trajectories.npy')), damped_trajectories)
    np.save(os.path.join(os.path.join(save_path, 'undamped_trajectories.npy')), undamped_trajectories)

def time_variation(args, new_steps=1200, num_videos=6, video_len=2, steps=120, delta_percentage=50):

    nsv_model, damped_model, undamped_model = prepare_Model(args, damping=0.0)
    undamped_ode = NeuralODE(undamped_model, solver='rk4').cuda()
    nsv_model.cuda()

    data_path = os.path.join(args.output_dir, args.dataset, 'variables', args.model_name, 'data.npy')
    data = np.load(data_path)

    data_max = np.max(data, axis=0)
    data_max_torch = torch.tensor(data_max).float().cuda().reshape(1,-1)
    data_min = np.min(data, axis=0)
    data_min_torch = torch.tensor(data_min).float().cuda().reshape(1,-1)
    nsv_range = np.sqrt(np.sum((data.max() - data.min())**2))

    save_path = os.path.join(args.output_dir, args.dataset, "downstream", args.model_name, f'timeVariation', f'videoLen_{video_len}_defaultSteps_{steps}_newSteps_{new_steps}')
    mkdir(save_path) #PATH

    start_values = []

    defaultStep_trajectories = []
    newStep_trajectories = []
    t_span = (video_len / steps * torch.arange(steps)).float()
    new_t_span = (video_len / new_steps * torch.arange(new_steps)).float()

    delta = .01 * delta_percentage * nsv_range
    perturbations = delta / num_videos
    rand_dir = torch.tensor(make_rand_vector(nsv_model.nsv_dim)).float().cuda()

    seed_everything(args.seed)
    mkdir(os.path.join(save_path, 'trajectories'))
    mkdir(os.path.join(save_path, 'all_newSteps'))
    mkdir(os.path.join(save_path, 'all_defaultSteps'))
    success = False

    start_values = []
    newStep_trajectories = []
    defaultStep_trajectories = []
    while success != True:

        rand_dir = torch.tensor(make_rand_vector(nsv_model.nsv_dim)).float().cuda()

        start_values_trial = []
        newStep_trajectories_trial = []
        defaultStep_trajectories_trial = []
        for i in tqdm(range(num_videos)):
            mkdir(os.path.join(save_path, f'defaultSteps_{i}'))
            mkdir(os.path.join(save_path, f'newSteps_{i}'))
            
            nsv_sample = damped_model.equilibrium + rand_dir * (i + 1) * perturbations
            nsv_sample = nsv_sample.unsqueeze(0)

            _, defaultStep_pred = undamped_ode(nsv_sample, t_span)
            _, newStep_pred = undamped_ode(nsv_sample, new_t_span)

            if torch.any(defaultStep_pred > data_max_torch) or torch.any(defaultStep_pred < data_min_torch) or torch.any(newStep_pred > data_max_torch) or torch.any(newStep_pred < data_min_torch):
                print("Escaped Boundary, Try again")
                break

            defaultStep_pred = defaultStep_pred.permute(1, 0, 2).squeeze()
            newStep_pred = newStep_pred.permute(1, 0, 2).squeeze()

            start_values_trial.append(nsv_sample)
            newStep_trajectories_trial.append(newStep_pred)
            defaultStep_trajectories_trial.append(defaultStep_pred)
            if i == num_videos - 1:
                success = True

        if success:
            for i in tqdm(range(num_videos)):
                newStep_pred = newStep_trajectories_trial[i]
                defaultStep_pred = defaultStep_trajectories_trial[i]
                start_value = start_values_trial[i]

                start_values.append(start_value.cpu().detach().numpy())
                defaultStep_trajectories.append(defaultStep_pred.cpu().detach().numpy())
                newStep_trajectories.append(newStep_pred.cpu().detach().numpy())

                plot_trajectory(defaultStep_pred.cpu().detach().numpy(), newStep_pred.cpu().detach().numpy(), save_path, i, pred_name='600 fps', target_name='60 fps')

                if "smooth" in nsv_model.name:
                    newStep_pred_output, _, _ = nsv_model.decoder(newStep_pred)
                    defaultStep_pred_output, _, _ = nsv_model.decoder(defaultStep_pred)
                else:
                    newStep_pred_output, _ = nsv_model.decoder(newStep_pred)
                    defaultStep_pred_output, _= nsv_model.decoder(defaultStep_pred)
                
                for idx in range(newStep_pred_output.shape[0]):
                    save_image(newStep_pred_output[idx, :, :, :128].cpu(), os.path.join(save_path,  f'newSteps_{i}/{idx}.png'), nrow=1)
                
                for idx in range(defaultStep_pred_output.shape[0]):
                    save_image(defaultStep_pred_output[idx, :, :, :128].cpu(), os.path.join(save_path,  f'defaultSteps_{i}/{idx}.png'), nrow=1)
                
                generate_video(save_path, f'newSteps_{i}', os.path.join(save_path, f'newSteps_{i}.mp4'), fps=120)
                generate_video(save_path, f'defaultSteps_{i}', os.path.join(save_path, f'defaultSteps_{i}.mp4'), fps=12)

                os.system(f'cp -r {save_path}/newSteps_{i} {save_path}/all_newSteps/{i}')
                os.system(f'cp -r {save_path}/defaultSteps_{i} {save_path}/all_defaultSteps/{i}')
    
    evaluator = Physics_Evaluator(args.dataset)
    marked_images_newSteps = evaluator.eval_physics(os.path.join(save_path, f'all_newSteps'), range(num_videos), new_steps, os.path.join(save_path, f'newSteps.npy'), return_marked=True)
    for vid_id, vid in enumerate(marked_images_newSteps):
        mkdir(os.path.join(save_path, f'newSteps_{vid_id}_marked'))
        for idx in range(len(vid)):
            cv2.imwrite(os.path.join(save_path, f'newSteps_{vid_id}_marked/{idx}_m.png'), vid[idx])

    marked_images_defaultSteps = evaluator.eval_physics(os.path.join(save_path, f'all_defaultSteps'), range(num_videos), steps, os.path.join(save_path, f'defaultSteps.npy'), return_marked=True)
    for vid_id, vid in enumerate(marked_images_defaultSteps):
        mkdir(os.path.join(save_path, f'defaultSteps_{vid_id}_marked'))
        for idx in range(len(vid)):
            cv2.imwrite(os.path.join(save_path, f'defaultSteps_{vid_id}_marked/{idx}_m.png'), vid[idx])
    
    newStep_trajectories = sort_trajectories(newStep_trajectories, damped_model.equilibrium.clone().detach().cpu().numpy())
    plot_with_gradField(newStep_trajectories, save_path, undamped_model, video_len / new_steps, 'newStep_trajectories.png', damped_model.equilibrium.clone().detach().cpu().numpy(), data_min, data_max)
    defaultStep_trajectories = sort_trajectories(defaultStep_trajectories, damped_model.equilibrium.clone().detach().cpu().numpy())
    plot_with_gradField(defaultStep_trajectories, save_path, undamped_model, video_len / steps, 'defaultStep_trajectories.png', damped_model.equilibrium.clone().detach().cpu().numpy(), data_min, data_max)

    new_start_step = new_steps//2
    new_image_files = [str(p+new_start_step)+'_m.png' for p in range(20)]
    default_start_step = steps//2
    default_image_files = [str(p+default_start_step)+'_m.png' for p in range(20)]

    for vid_id in range(num_videos):
        # Create full paths for images
        newStep_image_paths = [os.path.join(save_path, f'newSteps_{vid_id}_marked', img) for img in new_image_files]
        defaultStep_image_paths =  [os.path.join(save_path, f'defaultSteps_{vid_id}_marked', img) for img in default_image_files]

        newStep_output_path = os.path.join(save_path, f'startStep_{new_start_step}_newSteps_{vid_id}_marked.png')
        defaultStep_output_path = os.path.join(save_path, f'startStep_{default_start_step}_defaultSteps_{vid_id}_marked.png')

        concatenate_images_horizontally(newStep_image_paths, newStep_output_path)
        concatenate_images_horizontally(defaultStep_image_paths, defaultStep_output_path)


def plot_near_eq(trajectories, save_path, undamped_model, epsilon, equilibrium, dt):

    num_components = len(equilibrium)

    if num_components != 2:


        steps = trajectories[0].shape[0]

        fig_v = make_subplots(rows=num_components, cols=1)
        for i in range(num_components):
            fig_v.update_xaxes(range=[0,dt*(steps+1)],nticks=3,row=i)
            #fig_v.update_yaxes(title=f'<b>V{i+1}</b>', range=[equilibrium[i] - epsilon, equilibrium[i] + epsilon], tickmode='linear', tick0 = int(10*(equilibrium[i] - 0.7*epsilon))/10, dtick =  int(100*epsilon)/100, row=i+1,col=1)
            fig_v.update_yaxes(range=[equilibrium[i] - epsilon, equilibrium[i] + epsilon], tickmode='linear', tick0 = int(10*(equilibrium[i] - 0.7*epsilon))/10, dtick =  int(100*epsilon)/100, row=i+1,col=1)
        #fig_v.update_xaxes(title='<b>Time (s)</b>', range=[0,dt*(steps+1)],nticks=3,row=num_components)
        fig_v.update_xaxes(range=[0,dt*(steps+1)],nticks=3,row=num_components)
        fig_v.update_layout(title=None, showlegend=False)
        
        fig = make_subplots(rows=num_components-1, cols=num_components-1)
        fig.update_layout(showlegend=False)
        for t, traj in enumerate(trajectories):
            
            for i in range(num_components):
                fig_v.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=traj[:,i], line=dict(color=cols[t], width=3)), row=i+1,col=1)
        

            for i in range(num_components-1):
                for j in range(i+1, num_components):
                    fig.add_trace(go.Scatter(x=traj[:,i], y=traj[:,j], 
                                    mode='lines', line=dict(width=3, color=cols[t])), row=j, col=i+1)
                                
                    fig.update_xaxes(row=j, col=i+1, range=[equilibrium[i] - epsilon, equilibrium[i] + epsilon], tickmode='linear', tick0 = int(10*(equilibrium[i] - 0.7*epsilon))/10, dtick =  int(100*epsilon)/100, showticklabels= (j==num_components - 1))
                    fig.update_yaxes(row=j, col=i+1, range=[equilibrium[j] - epsilon, equilibrium[j] + epsilon], tickmode='linear', tick0 = int(10*(equilibrium[j] - 0.7*epsilon))/10, dtick =  int(100*epsilon)/100, showticklabels= (i==0))    

                    # if i == 0:
                    #     fig.update_yaxes(title_text=f"<b>V{j+1}</b>",row=j, col=i+1)
                    # if j == num_components - 1:
                    #     fig.update_xaxes(title_text=f"<b>V{i+1}</b>",row=j, col=i+1)

        update_figure_small(fig, True)  
        update_figure_small(fig_v, True)
    else:
        g = epsilon * np.mgrid[-1:1:15j,-1:1:15j]
        g = equilibrium + np.transpose(g, (1,2,0)).reshape(-1,2)

        grid_output = undamped_model(torch.from_numpy(g).type(torch.FloatTensor).cuda()).cpu().detach().numpy()

        scale = .005 if "spring_mass" in save_path else .01
        fig = ff.create_quiver(g[:,0], g[:,1], grid_output[:,0], grid_output[:,1], scale=scale, line=dict(color='grey'), line_width=3)
        # fig.update_layout(xaxis=dict(title='<b>V1</b>', range=[equilibrium[0] - epsilon, equilibrium[0] + epsilon], tickmode='linear', tick0 = int(10*(equilibrium[0] - epsilon) + 0.5)/10, dtick =  int(10*(equilibrium[0] + epsilon))/10 - int(10*(equilibrium[0] - epsilon) + 0.5)/10),
        #                         yaxis=dict(title='<b>V2</b>', range=[equilibrium[1] - epsilon, equilibrium[1] + epsilon], tickmode='linear', tick0 = int(10*(equilibrium[1] - epsilon) + 0.5)/10, dtick =  int(10*(equilibrium[1] + epsilon))/10 - int(10*(equilibrium[1] - epsilon) + 0.5)/10), 
        #                         width=700, height=640, showlegend=False)
        fig.update_layout(xaxis=dict(range=[equilibrium[0] - epsilon, equilibrium[0] + epsilon], tickmode='linear', tick0 = int(10*(equilibrium[0] - 0.7*epsilon))/10, dtick =  int(100*epsilon)/100),
                                yaxis=dict(range=[equilibrium[1] - epsilon, equilibrium[1] + epsilon], tickmode='linear', tick0 = int(10*(equilibrium[1] - 0.7*epsilon))/10, dtick =  int(100*epsilon)/100), 
                                width=700, height=640, showlegend=False)

        steps = trajectories[0].shape[0]

        fig_v = make_subplots(rows=2, cols=1)
        fig_v.update_xaxes(range=[0,dt*(steps+1)],nticks=3, row=1)
        # fig_v.update_yaxes(title='<b>V1</b>', range=[equilibrium[0] - epsilon, equilibrium[0] + epsilon],tickmode='linear', tick0 = int(10*(equilibrium[0] - epsilon) + 0.5)/10, dtick =  int(10*(equilibrium[0] + epsilon))/10 - int(10*(equilibrium[0] - epsilon) + 0.5)/10,row=1,col=1)
        fig_v.update_yaxes(range=[equilibrium[0] - epsilon, equilibrium[0] + epsilon],tickmode='linear', tick0 = int(10*(equilibrium[0] - 0.7*epsilon))/10, dtick =  int(100*epsilon)/100, row=1,col=1)
        #fig_v.update_yaxes(title='<b>V2</b>', range=[equilibrium[1] - epsilon, equilibrium[1] + epsilon],tickmode='linear', tick0 = int(10*(equilibrium[1] - 0.7*epsilon))/10, dtick =  int(100*epsilon)/100, row=2,col=1)
        fig_v.update_yaxes(range=[equilibrium[1] - epsilon, equilibrium[1] + epsilon],tickmode='linear', tick0 = int(10*(equilibrium[1] - 0.7*epsilon))/10, dtick =  int(100*epsilon)/100, row=2,col=1)
        #fig_v.update_xaxes(title='<b>Time (s)</b>', range=[0,dt*(steps+1)], nticks=3, row=2)
        fig_v.update_xaxes(range=[0,dt*(steps+1)], nticks=3, row=2)
        fig_v.update_layout(showlegend=False, width=700, height=700)

        for i, traj in enumerate(trajectories):
            fig.add_trace(go.Scatter(x=traj[:,0], y=traj[:,1], 
                                    mode='lines', line=dict(width=7, color=cols[i])))

            fig_v.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=traj[:,0], line=dict(color=cols[i], width=7), name=f"V1"), row=1,col=1)
            fig_v.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=traj[:,1], line=dict(color=cols[i], width=7), name=f"V2"), row=2,col=1)
    
        update_figure(fig, True)
        update_figure(fig_v, True)

    fig.write_image(os.path.join(save_path, 'trajectories.png'), scale=4)
    fig_v.write_image(os.path.join(save_path, 'V.png'), scale=4)


def near_eq(args, num_videos=len(cols), video_len=3, steps=180, delta_percentage=10, epsilon_percentage=10):

    nsv_model, damped_model, undamped_model = prepare_Model(args, damping=0.0)
    undamped_ode = NeuralODE(undamped_model, solver='rk4').cuda()
    nsv_model.cuda()

    save_path = os.path.join(args.output_dir, args.dataset, "downstream", args.model_name, 'near_eq', f'delta_{delta_percentage}_epsilon_{epsilon_percentage}_t_{steps}')
    mkdir(save_path) #PATH

    eq_path = os.path.join(args.output_dir, args.dataset, 'tasks', args.model_name, 'mlp_equilibrium', 'eq_points.npy')
    equilibriums = np.load(eq_path, allow_pickle=True).item()
    
    num_eqs = len(equilibriums["roots"])

    equilibrium = damped_model.equilibrium.cpu().numpy()
    # for i in range(num_eqs):
    #     if equilibriums['stabilities'][i] == "stable" and equilibriums['successes'][i] == True:
    #         equilibrium = equilibriums['roots'][i]
    #         break

    if not isinstance(equilibrium, np.ndarray):
        return

    begin_nsv = torch.FloatTensor(equilibrium).squeeze().cuda()
    print(begin_nsv)

    data_path = os.path.join(args.output_dir, args.dataset, 'variables', args.model_name, 'data.npy')
    data = np.load(data_path)

    data_max = np.max(data, axis=0)
    data_max_torch = torch.tensor(data_max).float().cuda().reshape(1,-1)
    data_min = np.min(data, axis=0)
    data_min_torch = torch.tensor(data_min).float().cuda().reshape(1,-1)
    nsv_range = np.sqrt(np.sum((data_max - data_min)**2))

    t_span = (video_len / steps * torch.arange(steps)).float().cuda()
    dt = video_len / steps

    seed_everything(args.seed)

    delta = .01 * delta_percentage * nsv_range
    epsilon = .01 * epsilon_percentage * nsv_range

    print(delta)
    perturbations = delta / num_videos

    success = False
    while success == False:
        trajectories = []
        start_values = []
        rand_dir = torch.tensor(make_rand_vector(nsv_model.nsv_dim)).float().cuda()
        
        for i in range(num_videos):

            mkdir(os.path.join(save_path, f'{i}'))

            nsv_sample = damped_model.equilibrium + rand_dir * (i + 1) * perturbations

            if torch.any(nsv_sample  > data_max_torch) or torch.any(nsv_sample < data_min_torch):
                print("Escaped Boundary, Try again")
                break

            initial_dist = torch.sqrt(torch.sum(((nsv_sample - begin_nsv)**2)))
            
            start_values.append(nsv_sample.cpu().detach().numpy())
            
            _, defaultStep_pred = undamped_ode(nsv_sample, t_span)

            if torch.any(defaultStep_pred > data_max_torch) or torch.any(defaultStep_pred < data_min_torch):
                print("Escaped Boundary, Try again")
                break
            else:

                trajectories.append(defaultStep_pred.cpu().detach().numpy())

                if "smooth" in nsv_model.name:
                    defaultStep_pred_output, _, _ = nsv_model.decoder(defaultStep_pred)
                else:
                    defaultStep_pred_output, _ = nsv_model.decoder(defaultStep_pred)

                for idx in range(defaultStep_pred_output.shape[0]):
                    save_image(defaultStep_pred_output[idx, :, :, :128].cpu(), os.path.join(save_path,  f'{i}/{idx}.png'), nrow=1)
                
                generate_video(save_path, f'{i}', os.path.join(save_path, f'{i}.mp4'), fps=60, delete_after=True)
                
                if i == num_videos - 1:
                    success = True

    np.save(os.path.join(os.path.join(save_path, 'delta_epsilon.npy')), np.array([(delta_percentage, delta),(epsilon_percentage, epsilon)]))
    np.save(os.path.join(os.path.join(save_path, 'start_values.npy')), np.array(start_values))
    np.save(os.path.join(os.path.join(save_path, 'trajectories.npy')), np.array(trajectories))

    trajectories = sort_trajectories(trajectories, equilibrium)
    
    plot_near_eq(trajectories, save_path, undamped_model, epsilon, equilibrium, dt)



def detect_chaos(args, specific=False):

    if args.dataset not in ['spring_mass', 'single_pendulum', 'double_pendulum']:
        return

    grid_size=int(10000**(1/get_experiment_dim(args.dataset, args.seed)))
    
    if 'smooth' in args.model_name:
        model = SmoothNSVAutoencoder(**args)
    elif 'base' in args.model_name:
        model = NSVAutoencoder(**args)
    
    weight_path = get_weightPath(args,  False)
    net = VisDynamicsModel.load_from_checkpoint(weight_path, model=model, **args)
    net.eval()

    output_path = os.path.join(os.getcwd(), args.output_dir, args.dataset)
    model_name = create_name(args)

    trainer = Trainer(devices=args.num_gpus,
                      deterministic=True,
                      accelerator='gpu',
                      default_root_dir=output_path,
                      **get_validArgs(Trainer, args))
    
    if 'smooth' in model_name:
        test_dataset = NeuralPhysSmoothDataset(data_filepath=args.data_filepath,
                                                        flag='test',
                                                        seed=args.seed,
                                                        object_name=f'{args.dataset}_long_sequence')
    elif 'base' in args.model_name:
        test_dataset = NeuralPhysDataset(data_filepath=args.data_filepath,
                                                        flag='test',
                                                        seed=args.seed,
                                                        object_name=f'{args.dataset}_long_sequence')
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    net.test_mode = "chaos"
    net.pred_log_name = 'predictions_long'
    net.var_log_name = 'variables_long'
    net.task_log_dir = 'tasks_long'

    result = trainer.test(net, test_dataloader)

    ids_test = np.array(net.all_filepaths)
    nsv_test = np.array(net.all_refine_latents)

    data_max, data_min = nsv_test.max(axis=0), nsv_test.min(axis=0)
    nsv_range = np.sqrt(np.sum((data_max - data_min)**2))
    data_full_max = np.stack([np.abs(data_min), data_max], axis=0).max(axis=0)
    #print(nsv_range)

    def trajectories_from_data_ids(ids, nsv):
        id2index = {tuple(id): i for i, id in enumerate(ids)}
        trajectories = collections.defaultdict(list)

        for id in sorted(id2index.keys()):
            i = id2index[id]
            trajectories[id[0]].append(nsv[i])

        for key, value in trajectories.items():
            trajectories[key] = np.array(value)
        return trajectories

    trajectories = trajectories_from_data_ids(ids_test, nsv_test)
    
    save_path_base = os.path.join(args.output_dir, args.dataset, "downstream", model_name, 'chaos_specific' if specific else 'chaos')
    mkdir(save_path_base)

    occupancy_path = os.path.join(save_path_base, 'occupancy')
    mkdir(occupancy_path)
    traj_ratio_visited_cells = {}
    total_traj_ratio_visited_cells = {}

    lengths = []
    mkdir(os.path.join(occupancy_path, 'visit_ratio'))
    mkdir(os.path.join(occupancy_path, '3d_plots'))

    all_traj_fig = go.Figure()
    dt = 1/60 
    for i in trajectories.keys():

        visited_cells, ratio_visited_cells = calculate_trajectory_occupancy(trajectories[i], data_max, data_min, grid_size)
        
        traj_ratio_visited_cells[i] = ratio_visited_cells
        total_traj_ratio_visited_cells[i] = ratio_visited_cells[-1]
        steps = ratio_visited_cells.shape[0]
        lengths.append(steps)
        t = np.linspace(dt,dt*(steps), steps)

        all_traj_fig.add_trace(go.Scatter(x=t, y=ratio_visited_cells, mode='lines', line=dict(color=cols[0], width=5),showlegend=False))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=ratio_visited_cells, mode='lines', line=dict(color=cols[0], width=5),showlegend=False))
        #fig.update_yaxes(title=f"<b>Visited %</b>", tickmode='auto')
        fig.update_yaxes(tickmode='auto')
        #fig.update_xaxes(title="<b>Time (s)</b>", range=[dt,dt*(steps)], nticks=6)
        fig.update_xaxes(range=[dt,dt*(steps)], nticks=6)
        update_figure(fig, True)
        fig.write_image(os.path.join(occupancy_path, 'visit_ratio', f"{i}.png"), scale=4)

        traj = trajectories[i]
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=traj[:,0], y=traj[:,1], z=traj[:,2], mode='lines',
                        line=dict(color=cols[0], width=5)
                    ))
        
        fig.add_trace(go.Cone(sizeref=5, anchor='tip', showscale=False, colorscale=[[0, cols[0]], [1,cols[0]]], x=traj[0:1,0], y=traj[0:1,1], z=traj[0:1,2], u=traj[1:2,0]-traj[0:1,0], v=traj[1:2,1]-traj[0:1,1], w=traj[1:2,2]-traj[0:1,2]))

        # fig.update_layout(#title=f'video no. {vid_idx}', 
        #                     scene=dict(aspectratio=dict(x=1, y=1, z=1),
        #                     xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
        #                     yaxis=dict(title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
        #                     zaxis=dict(title='<b>V3</b>', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10)), showlegend=False)
        fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
                            xaxis=dict(title='', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                            yaxis=dict(title='', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                            zaxis=dict(title='', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10)), showlegend=False)
            
        update_figure_3d(fig, True)
        fig.write_image(os.path.join(occupancy_path, '3d_plots', f"{i}.png"), scale=4)
    
    # all_traj_fig.update_yaxes(title=f"<b>Visited %</b>", tickmode='auto')
    all_traj_fig.update_yaxes( tickmode='auto')
    #all_traj_fig.update_xaxes(title="<b>Time (s)</b>", range=[dt,dt*(steps)], nticks=6)
    all_traj_fig.update_xaxes(range=[dt,dt*(steps)], nticks=6)
    all_traj_fig.update_xaxes(range=[dt,dt*(steps)], nticks=6)
    update_figure(all_traj_fig, False)
    all_traj_fig.write_image(os.path.join(occupancy_path, 'visit_ratio', f"all_traj.png"), scale=4)
    
    np.save(os.path.join(occupancy_path, 'trajectories.npy'), trajectories)
    np.save(os.path.join(occupancy_path, 'traj_ratio_visited_cells.npy'), traj_ratio_visited_cells)
    np.save(os.path.join(occupancy_path, 'total_traj_ratio_visited_cells.npy'), total_traj_ratio_visited_cells)
    
    if 'double_pendulum' in args.dataset:
        lengths = np.array(lengths)
        kmeans = KMeans(n_clusters=2)
        scaled_total_traj_ratio_visited_cells = np.array(list(total_ratio / (dt*lengths[i]) for i, total_ratio in total_traj_ratio_visited_cells.items()))

        kmeans.fit(scaled_total_traj_ratio_visited_cells.reshape(-1,1))
        print("Cluster Centers: ", kmeans.cluster_centers_)
        separating_threshold = np.mean(kmeans.cluster_centers_)
        print("Separating Threshold: ", separating_threshold)
        
        count_chaotic = 0
        count_periodic = 0
        traj_chaotic = {}
        for i in trajectories.keys():
            
            traj_chaotic[i] = is_chaotic(scaled_total_traj_ratio_visited_cells[i], threshold=separating_threshold)
            #label = kmeans.predict(np.array([scaled_total_traj_ratio_visited_cells[i]]).reshape(-1,1))[0]

            if not traj_chaotic[i] == 1:
                count_periodic += 1
            else:
                count_chaotic += 1
        assert(count_chaotic + count_periodic == len(traj_chaotic))
        print(f"Percentage Trajectory Chaotic: {100 * count_chaotic/len(traj_chaotic)} %")
        print(f"Percentage Trajectory Periodic: {100 * count_periodic/len(traj_chaotic)} %")

        np.save(os.path.join(occupancy_path, 'traj_chaotic.npy'), traj_chaotic)

    ini_err = {}
    pred_len = 600
    if not specific:
        for i1 in range(99):
            for i2 in range(i1+1, 100):
                for t1 in range(50):
                    for t2 in range(50):
                        sel = (i1, i2, t1, t2)
                        ini_err[sel] = np.linalg.norm(trajectories[i1][t1] - trajectories[i2][t2])
        sorted_ini_err = list(sorted(ini_err.items(), key=lambda x:x[1], reverse=False))
    else:
        sel = (41,46,35,34)
        ini_err[sel] = np.linalg.norm(trajectories[sel[0]][sel[2]] - trajectories[sel[1]][sel[3]])
        sel = (58,60,4,30)
        ini_err[sel] = np.linalg.norm(trajectories[sel[0]][sel[2]] - trajectories[sel[1]][sel[3]])
        sorted_ini_err = list(sorted(ini_err.items(), key=lambda x:x[1], reverse=False))

    comparison_path = os.path.join(save_path_base, 'comparisons')
    mkdir(comparison_path)
    
    visited_pairs = set()
    trajectory_pairs = {}
    for (i1, i2, t1, t2), err in sorted_ini_err:
        
        if not specific and err > 0.01 * nsv_range:
            break
        
        if (i1, i2) not in visited_pairs:
            save_name = f'ini {err} {i1}_{t1} vs {i2}_{t2}'

            trajectory_pairs[(i1, i2)]= np.stack([trajectories[i1][t1:t1+pred_len], trajectories[i2][t2:t2+pred_len]], axis=0)
            visualize_trajectory_chaos(trajectories[i1][t1:t1+pred_len], trajectories[i2][t2:t2+pred_len], comparison_path, save_name, data_max, data_min, grid_size, dt)

            #plot_occupancy(trajectories[i1][t1:t1+pred_len], trajectories[i2][t2:t2+pred_len], comparison_path, save_name, data_max, data_min, grid_size)

            visited_pairs.add((i1, i2))

    np.save(os.path.join(comparison_path, 'trajectory_pairs.npy'), trajectory_pairs, allow_pickle=True)


def is_chaotic(traj_ratio_visited_cells_max, threshold):

    return 1 if traj_ratio_visited_cells_max >= threshold else 0
    
def calculate_trajectory_occupancy(traj, data_max, data_min, N, window=60):

    data_range = data_max - data_min
    #print(data_range/N)
    def get_cell_index(state):
        """Convert state in [-1, 1] domain to grid index."""
        return np.floor((state - data_min) * (N-1) / data_range).astype(int)
    
    visited_cells = set([tuple(get_cell_index(traj[0,:]))])
    num_visited_cells = [len(visited_cells)]

    steps = traj.shape[0]
    for i in range(steps-1):

        start_state = traj[i,:] 
        end_state = traj[i+1,:]

        start_cell = get_cell_index(start_state)
        end_cell = get_cell_index(end_state)

        d = end_cell-start_cell
        max_d = np.max(np.abs(d))
        s = d/max_d if max_d != 0 else 0

        cur_cell = np.copy(start_cell)
        for j in range(max_d):
            visited_cells.add(tuple(np.floor(cur_cell).astype(int)))
            cur_cell = cur_cell + s
        visited_cells.add(tuple(end_cell))

        num_visited_cells.append(len(visited_cells))
    
    ratio_visited_cells =  100 * np.array(num_visited_cells) / N**(data_max.shape[-1])

    return visited_cells, ratio_visited_cells
    
def plot_trajectory_pair(traj1, traj2, save_path, save_name, data_full_max, dt):

    if traj1.shape[1] == 4 or traj1.shape[1] == 3:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=traj1[:,0], y=traj1[:,1], z=traj1[:,2], mode='lines',
                        line=dict(color=cols[0], width=5)
                    ))
        fig.add_trace(go.Scatter3d(x=traj2[:,0], y=traj2[:,1], z=traj2[:,2], mode='lines',
                        line=dict(color=cols[1], width=5)
                    ))
        
        fig.add_trace(go.Cone(sizeref=5, anchor='tip', showscale=False, colorscale=[[0, cols[0]], [1,cols[0]]], x=traj1[0:1,0], y=traj1[0:1,1], z=traj1[0:1,2], u=traj1[1:2,0]-traj1[0:1,0], v=traj1[1:2,1]-traj1[0:1,1], w=traj1[1:2,2]-traj1[0:1,2]))
        fig.add_trace(go.Cone(sizeref=5, anchor='tip',showscale=False, colorscale=[[0, cols[1]], [1,cols[1]]], x=traj2[0:1,0], y=traj2[0:1,1], z=traj2[0:1,2], u=traj2[1:2,0]-traj2[0:1,0], v=traj2[1:2,1]-traj2[0:1,1], w=traj2[1:2,2]-traj2[0:1,2]))

        # fig.update_layout(#title=f'video no. {vid_idx}', 
        #                     scene=dict(aspectratio=dict(x=1, y=1, z=1),
        #                     xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
        #                     yaxis=dict(title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
        #                     zaxis=dict(title='<b>V3</b>', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10)), showlegend=False)
        fig.update_layout(#title=f'video no. {vid_idx}', 
                            scene=dict(aspectratio=dict(x=1, y=1, z=1),
                            xaxis=dict(title='', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                            yaxis=dict(title='', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                            zaxis=dict(title='', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10)), showlegend=False)
            
        update_figure_3d(fig, True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=traj1[:,0], y=traj1[:,1], mode='lines',
                        line=dict(color=cols[0], width=4)
                    ))
        fig.add_trace(go.Scatter(x=traj2[:,0], y=traj2[:,1], mode='lines',
                        line=dict(color=cols[1], width=4)
                    ))
        
        fig.add_trace(go.Scatter( x=traj1[0:1,0], y=traj1[0:1,1], marker=dict(
                symbol="arrow",
                size=15,
                angle=55,
                color=cols[0]
            )))
        fig.add_trace(go.Scatter( x=traj2[0:1,0], y=traj2[0:1,1], marker=dict(
                symbol="arrow",
                size=15,
                angle=35,
                color=cols[1]
            )))
        # fig.update_layout(xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
        #                 yaxis=dict(scaleanchor="x", scaleratio=1, title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
        #                 showlegend=False)
        fig.update_layout(xaxis=dict(range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
                        yaxis=dict(scaleanchor="x", scaleratio=1, range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
                        showlegend=False)
        update_figure(fig, True)

    fig.write_image(os.path.join(save_path, f'{save_name}.png'), scale=4)

    steps = traj1.shape[0]
    fig = make_subplots(rows=4, cols=1)
    for i in range(traj1.shape[1]):
        fig.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=traj1[:,i], mode='lines', line=dict(color=cols[i], width=5),showlegend=False), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=traj2[:,i], mode='lines', line=dict(color=cols[i], dash="dot", width=5),showlegend=False), row=i+1, col=1)
        #fig.update_yaxes(row=i+1, col=1, title=f"<b>V{i+1}</b>", range=[-1.1*data_full_max[i], 1.1*data_full_max[i]], tickmode='linear', tick0 = int(10*data_full_max[i])/10, dtick=int(10*data_full_max[i])/10)
        fig.update_yaxes(row=i+1, col=1, range=[-1.1*data_full_max[i], 1.1*data_full_max[i]], tickmode='linear', tick0 = int(10*data_full_max[i])/10, dtick=int(10*data_full_max[i])/10)
        fig.update_xaxes(row=i+1, col=1, range=[0,dt*(steps)], nticks=6) #, showticklabels=(i == traj1.shape[1] - 1))
    # fig.update_xaxes(row=4, col=1, title="<b>Time (s)</b>")
    update_figure_small(fig, True)
    fig.write_image(os.path.join(save_path, f"{save_name}_p.png"), scale=4)

    return



def plot_perturbation(traj1, traj2, save_path, save_name, dt):
    dt = 1/60 
    perturbation = np.sqrt(np.sum((traj1 - traj2)**2, axis=1))
    steps = perturbation.shape[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=perturbation, mode='lines', line=dict(color=cols[0], width=5),showlegend=False))
    # fig.update_yaxes(title=f"<b>Difference</b>", tickmode='auto')
    fig.update_yaxes(tickmode='auto')
    # fig.update_xaxes(title="<b>Time (s)</b>", range=[0,dt*(steps)], nticks=6)
    fig.update_xaxes(range=[0,dt*(steps)], nticks=6)
    update_figure(fig, True)
    fig.write_image(os.path.join(save_path, f"{save_name}_e.png"), scale=4)

    return

def plot_occupancy(traj1, traj2, save_path, save_name, data_max, data_min, N, dt):
    
    visited_cells_1, ratio_visited_cells_1 = calculate_trajectory_occupancy(traj1, data_max, data_min, N)
    visited_cells_2, ratio_visited_cells_2 = calculate_trajectory_occupancy(traj2, data_max, data_min, N)


    steps = ratio_visited_cells_1.shape[0]

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=np.linspace(dt,dt*(steps), steps), y=ratio_visited_cells_1, mode='lines', line=dict(color=cols[0], width=5),showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.linspace(dt,dt*(steps), steps), y=ratio_visited_cells_2, mode='lines', line=dict(color=cols[1], width=5),showlegend=False), row=2, col=1)
    # fig.update_yaxes(row=1, col=1, title=f"<b>Domain Visited %</b>", tickmode='auto')
    fig.update_yaxes(row=1, col=1, tickmode='auto')
    # fig.update_yaxes(row=2, col=1, title=f"<b>Domain Visited %</b>", tickmode='auto')
    fig.update_yaxes(row=2, col=1, tickmode='auto')
    fig.update_xaxes(row=1, col=1, range=[dt,dt*(steps)], nticks=6) #, showticklabels=False)
    #fig.update_xaxes(row=2, col=1, range=[dt,dt*(steps)], title="<b>Time (s)</b>", nticks=6)
    fig.update_xaxes(row=2, col=1, range=[dt,dt*(steps)], nticks=6)
    update_figure_small(fig, True)
    fig.write_image(os.path.join(save_path, f"{save_name}_occupancy.png"), scale=4)

    return 


def visualize_trajectory_chaos(traj1, traj2, save_path, save_name, data_max, data_min, grid_size, dt):

    
    data_full_max = np.stack([np.abs(data_min), data_max], axis=0).max(axis=0)

    # if traj1.shape[1] == 4 or traj1.shape[1] == 3:
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter3d(x=traj1[:,0], y=traj1[:,1], z=traj1[:,2], mode='lines',
    #                     line=dict(color=cols[0], width=5)
    #                 ))
    #     fig.add_trace(go.Scatter3d(x=traj2[:,0], y=traj2[:,1], z=traj2[:,2], mode='lines',
    #                     line=dict(color=cols[1], width=5)
    #                 ))
        
    #     fig.add_trace(go.Cone(sizeref=5, anchor='tip', showscale=False, colorscale=[[0, cols[0]], [1,cols[0]]], x=traj1[0:1,0], y=traj1[0:1,1], z=traj1[0:1,2], u=traj1[1:2,0]-traj1[0:1,0], v=traj1[1:2,1]-traj1[0:1,1], w=traj1[1:2,2]-traj1[0:1,2]))
    #     fig.add_trace(go.Cone(sizeref=5, anchor='tip',showscale=False, colorscale=[[0, cols[1]], [1,cols[1]]], x=traj2[0:1,0], y=traj2[0:1,1], z=traj2[0:1,2], u=traj2[1:2,0]-traj2[0:1,0], v=traj2[1:2,1]-traj2[0:1,1], w=traj2[1:2,2]-traj2[0:1,2]))

    #     # fig.update_layout(#title=f'video no. {vid_idx}', 
    #     #                     scene=dict(aspectratio=dict(x=1, y=1, z=1),
    #     #                     xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
    #     #                     yaxis=dict(title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
    #     #                     zaxis=dict(title='<b>V3</b>', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10)), showlegend=False)
    #     fig.update_layout(#title=f'video no. {vid_idx}', 
    #                         scene=dict(aspectratio=dict(x=1, y=1, z=1),
    #                         xaxis=dict(title='', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
    #                         yaxis=dict(title='', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
    #                         zaxis=dict(title='', range=[-1.1*data_full_max[2], 1.1*data_full_max[2]], tickmode='linear', tick0 = int(10*data_full_max[2])/10, dtick=int(10*data_full_max[2])/10)), showlegend=False)
            
    #     update_figure_3d(fig, True)
    # else:
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=traj1[:,0], y=traj1[:,1], mode='lines',
    #                     line=dict(color=cols[0], width=4)
    #                 ))
    #     fig.add_trace(go.Scatter(x=traj2[:,0], y=traj2[:,1], mode='lines',
    #                     line=dict(color=cols[1], width=4)
    #                 ))
        
    #     fig.add_trace(go.Scatter( x=traj1[0:1,0], y=traj1[0:1,1], marker=dict(
    #             symbol="arrow",
    #             size=15,
    #             angle=55,
    #             color=cols[0]
    #         )))
    #     fig.add_trace(go.Scatter( x=traj2[0:1,0], y=traj2[0:1,1], marker=dict(
    #             symbol="arrow",
    #             size=15,
    #             angle=35,
    #             color=cols[1]
    #         )))
    #     # fig.update_layout(xaxis=dict(title='<b>V1</b>', range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
    #     #                 yaxis=dict(scaleanchor="x", scaleratio=1, title='<b>V2</b>', range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
    #     #                 showlegend=False)
    #     fig.update_layout(xaxis=dict(range=[-1.1*data_full_max[0], 1.1*data_full_max[0]], tickmode='linear', tick0 = int(10*data_full_max[0])/10, dtick=int(10*data_full_max[0])/10),
    #                     yaxis=dict(scaleanchor="x", scaleratio=1, range=[-1.1*data_full_max[1], 1.1*data_full_max[1]], tickmode='linear', tick0 = int(10*data_full_max[1])/10, dtick=int(10*data_full_max[1])/10),
    #                     showlegend=False)
    #     update_figure(fig, True)

    # fig.write_image(os.path.join(save_path, f'{save_name}.png'), scale=4)

    # steps = traj1.shape[0]
    # fig = make_subplots(rows=4, cols=1)
    # for i in range(traj1.shape[1]):
    #     fig.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=traj1[:,i], mode='lines', line=dict(color=cols[i], width=5),showlegend=False), row=i+1, col=1)
    #     fig.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=traj2[:,i], mode='lines', line=dict(color=cols[i], dash="dot", width=5),showlegend=False), row=i+1, col=1)
    #     #fig.update_yaxes(row=i+1, col=1, title=f"<b>V{i+1}</b>", range=[-1.1*data_full_max[i], 1.1*data_full_max[i]], tickmode='linear', tick0 = int(10*data_full_max[i])/10, dtick=int(10*data_full_max[i])/10)
    #     fig.update_yaxes(row=i+1, col=1, range=[-1.1*data_full_max[i], 1.1*data_full_max[i]], tickmode='linear', tick0 = int(10*data_full_max[i])/10, dtick=int(10*data_full_max[i])/10)
    #     fig.update_xaxes(row=i+1, col=1, range=[0,dt*(steps)], nticks=6) #, showticklabels=(i == traj1.shape[1] - 1))
    # # fig.update_xaxes(row=4, col=1, title="<b>Time (s)</b>")
    # update_figure_small(fig, True)
    # fig.write_image(os.path.join(save_path, f"{save_name}_p.png"), scale=4)

    plot_trajectory_pair(traj1, traj2, save_path, save_name, data_full_max, dt)

    # perturbation = np.sqrt(np.sum((traj1 - traj2)**2, axis=1))
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=np.linspace(0,dt*(steps-1), steps), y=perturbation, mode='lines', line=dict(color=cols[0], width=5),showlegend=False))
    # # fig.update_yaxes(title=f"<b>Difference</b>", tickmode='auto')
    # fig.update_yaxes(tickmode='auto')
    # # fig.update_xaxes(title="<b>Time (s)</b>", range=[0,dt*(steps)], nticks=6)
    # fig.update_xaxes(range=[0,dt*(steps)], nticks=6)
    # update_figure(fig, True)
    # fig.write_image(os.path.join(save_path, f"{save_name}_e.png"), scale=4)

    plot_perturbation(traj1, traj2, save_path, save_name, dt)

    plot_occupancy(traj1, traj2, save_path, save_name, data_max, data_min, grid_size, dt)


def sort_trajectories(trajectories, equilibrium):

    trajectories.sort(key=lambda x : np.mean(np.sum((x - equilibrium)**2, axis=1), axis=0), reverse=True)

    return trajectories



def main():

    parser = argparse.ArgumentParser(description='Smooth Neural State Variable Downstream Tasks')

    parser.add_argument('-delta', 
                    type=float, default=35)
    parser.add_argument('-epsilon', 
                    type=float, default=10)
    parser.add_argument('-task', help='task type',
                    type=str, required=True)
    parser.add_argument('-config', help='config file path',
                    type=str, required=True)

    script_args = parser.parse_args()

    cfg = load_config(filepath=script_args.config)
    pprint.pprint(cfg)

    args = munchify(cfg)

    if script_args.task ==  "damping":
        return damping(args, damping=4, delta_percentage=script_args.delta)
    if script_args.task ==  "dt":
        return time_variation(args, delta_percentage=script_args.delta)
    if script_args.task == "chaos":
        return detect_chaos(args)
    if script_args.task == "chaos_specific":
        return detect_chaos(args, True)
    if script_args.task == "near_eq":
        return near_eq(args,video_len=2, steps=120, delta_percentage=script_args.delta, epsilon_percentage=script_args.epsilon)
    if script_args.task == "all":
        tmp = args.model_name
        damping_constant = 4 if 'double_pendulum' in args.dataset else 1
        damping(args, damping=damping_constant, delta_percentage=script_args.delta)

        args.model_name = tmp
        time_variation(args,  delta_percentage=script_args.delta)

        args.model_name = tmp
        near_eq_delta = 0.5 if 'double_pendulum' in args.dataset else 1
        near_eq(args,video_len=2, steps=120, delta_percentage=near_eq_delta, epsilon_percentage=1.5)



if __name__ == '__main__':
    
    main()