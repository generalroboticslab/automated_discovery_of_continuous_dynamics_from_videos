import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from collections import OrderedDict

from utils.misc import *
from utils.show import *
from .analysis import Physics_Evaluator


def pred(net, args, step = 3):

    base_model_name = args.base_model_name if hasattr(args, "base_model_name") else f"base_{args.seed}"

    methods = []

    print('Model Rollout')
    model_rollout(net, args)
    methods.append('model_rollout')
    

    success = model_rollout_hybrid(net, args, step)
    if success:
        methods.append(f'hybrid_rollout_{step}')

    if 'base' in net.model.name:
        base_log_dir = None
    else:
        base_log_dir = os.path.join(args.output_dir, args.dataset, 'predictions_long_term', base_model_name)
    

    print('Calculating Physics Error')
    analyze_trajectories(args.seed, args.dataset, os.path.join(args.output_dir, args.dataset, 'predictions_long_term', net.model.name), methods, base_log_dir, args.data_filepath)



def evaluate_physics_directory(dataset, long_term_folder, path_nums):

    physics_variable_path = os.path.join(long_term_folder, 'phys_vars.npy')

    total_num_frames = len(os.listdir(os.path.join(long_term_folder, str(path_nums[0]))))

    print("Evaluating Physics")
    evaluator = Physics_Evaluator(dataset)
    evaluator.eval_physics(long_term_folder, path_nums, total_num_frames, physics_variable_path)

def load_data_physics(data_filepath, dataset, phys_vars_list, seed):

    data_filepath_base = os.path.join(data_filepath, dataset)
    with open(os.path.join(data_filepath_base, 'datainfo', f'data_split_dict_{seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    vid_ids = seq_dict['test']

    phys = np.load(os.path.join(data_filepath, dataset, 'phys_vars.npy'), allow_pickle=True).item()
    seen = set()
    for p_var in phys_vars_list:
        if p_var in seen:
            break
        phys[p_var] = phys[p_var][vid_ids]
        seen.add(p_var)
    return phys

def analyze_trajectories(seed, dataset, log_dir, methods, baseline_log_dir, data_filepath='data'):

    if dataset == "fire":
        return
    
    dt = 1/60 if dataset != 'cylindrical_flow' else .02
        
    evaluator = Physics_Evaluator(dataset)
    phys_vars_list = evaluator.get_phys_vars(False)

    mkdir(os.path.join(log_dir, 'plots'))
    fig, axs = plt.subplots(1,len(phys_vars_list) + 1, layout='constrained', figsize=(8*(len(phys_vars_list)+1), 6))

    figs = [go.Figure() for _ in range(len(phys_vars_list) + 1)]

    gt = load_data_physics(data_filepath, dataset, evaluator.get_phys_vars(True), seed)

    losses = {}
    for i, method in enumerate(methods):

        losses[method] = {}
        phys_variables = np.load(os.path.join(log_dir, method, 'phys_vars.npy'), allow_pickle=True).item()
        
        losses[method]['reject'] = phys_variables['reject'].copy()
        losses[method]['reject_data'] = gt['reject'].copy()

        pred_len = losses[method]['reject'].shape[1]

        reject_ratio_mean = scale_reject_ratio(pred_len, phys_variables['reject'],  gt['reject'])
        
        axs[0].plot(range(pred_len), reject_ratio_mean, label=method)
        figs[0].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=reject_ratio_mean, line=dict(color=cols[i], width=4), name=f'Smooth {method}'))
        losses[method]['reject_ratio'] = reject_ratio_mean

        for var in phys_vars_list:

            losses[method][var] = []

            for traj_i, traj in enumerate(phys_variables[var]):
                
                traj_loss = []
                
                for i, val in enumerate(traj):
                    if phys_variables['reject'][traj_i][i]:
                        traj_loss.append(np.nan)
                    else:
                        if var in ['theta', 'theta_1', 'theta_2']:
                            diff = calc_theta_diff(gt[var][traj_i][i],val)
                        else:
                            diff = abs(gt[var][traj_i][i] - val)#**2)
                        
                        if 'theta' in var:
                            diff = diff * 180 / np.pi
                            
                        traj_loss.append(diff)
                
                losses[method][var].append(traj_loss)

            losses[method][var] = np.array(losses[method][var])
    
    np.save(os.path.join(log_dir, 'losses.npy'), losses)

    if baseline_log_dir != None and os.path.exists(os.path.join(baseline_log_dir, 'losses.npy')):
        base_line_losses = np.load(os.path.join(baseline_log_dir, 'losses.npy'),allow_pickle=True).item()
        baseline_methods = ['hybrid_rollout_3', 'model_rollout']
         
        for i, baseline_method in enumerate(baseline_methods):
            #plt.plot(range(len(base_line_losses[baseline_method]['reject_ratio'])), base_line_losses[baseline_method]['reject_ratio'], label="baseline_"+baseline_method)
            axs[0].plot(range(len(base_line_losses[baseline_method]['reject_ratio'])), base_line_losses[baseline_method]['reject_ratio'], label="baseline_"+baseline_method)
            figs[0].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=base_line_losses[baseline_method]['reject_ratio'], line=dict(color=cols[i+ len(methods)], width=4), name=f'Non-Smooth {baseline_method}'))
    else:
        base_line_losses = None
        baseline_methods = ['hybrid_rollout_3', 'model_rollout']

    axs[0].set_title(' Reject Ratio')
    axs[0].set_xlabel("time step")
    axs[0].set_ylabel("Reject Ratio")
    axs[0].set_ylim(-.01,1.05)

    update_figure(figs[0])
    figs[0].update_layout(title='Reject Ratio', 
                            xaxis=dict(title='t', range=[0, dt*(pred_len+1)]),
                            yaxis=dict(title='Reject Ratio', range=[-.01,1.05]), 
                            showlegend=False)

    phys_error_mean = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}
    phys_error_sem = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}

    pred_len = 60 if dataset != 'cylindrical_flow' else 200
    for i, var in  enumerate(phys_vars_list):
        #plt.figure()

        for j, method in enumerate(methods):
            for p in range(pred_len):
                error_p = losses[method][var][:, p]
                error_p = error_p[~np.isnan(error_p)]
                error_p = remove_outlier(error_p)
                if error_p.size > 1:
                    phys_error_mean[var][method][p] = np.mean(error_p)
                    phys_error_sem[var][method][p] = stats.sem(error_p)
                else:
                    phys_error_mean[var][method][p] = np.nan
                    phys_error_sem[var][method][p] = np.nan
            
            axs[i+1].plot(range(pred_len), phys_error_mean[var][method], label=method)
            figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=phys_error_mean[var][method], line=dict(color=cols[j], width=4), name=f'Smooth {method}'))

        if base_line_losses != None:
            for j, method in enumerate(baseline_methods):
                baseline_loss = np.zeros(pred_len)
                for p in range(pred_len):
                    error_p = base_line_losses[method][var][:, p]
                    error_p = error_p[~np.isnan(error_p)]
                    error_p = remove_outlier(error_p)
                    if error_p.size > 1:
                        baseline_loss[p] = np.mean(error_p)
                    else:
                        baseline_loss[p] = np.nan
                
                axs[i+1].plot(range(pred_len), baseline_loss, label=f'baseline_{method}')
                figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=baseline_loss, line=dict(color=cols[j + len(methods)], width=4), name=f'Non-Smooth {method}'))

        axs[i+1].set_title(f'{var} error (L1)')
        axs[i+1].set_xlabel("time step")
        axs[i+1].set_ylabel("L1 error (ignoring nan)")
        
        update_figure(figs[i+1])
        figs[i+1].update_layout(title=f'{var} error (L1)', 
                            xaxis=dict(title='t', range=[0, dt*(pred_len+1)]),
                            yaxis=dict(title="L1 error (ignoring nan)"), 
                            showlegend=False)
    
    lines,labels = axs[0].get_legend_handles_labels()
    fig.legend(lines,labels , loc='outside lower center', fontsize="20", ncol = len(baseline_methods)+2)
    fig.savefig(os.path.join(log_dir, 'plots', f'all.png'))

    phys_vars_list = evaluator.get_phys_vars(True)
    for i, f in enumerate(figs):
        f.write_image(os.path.join(log_dir, 'plots', f'{phys_vars_list[i]}.png'), scale=4)
    
    fig = go.Figure()
    for i, method in enumerate(methods):
        fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'Smooth {method}', line=dict(color=cols[i])))
    if base_line_losses != None:
        for i, method in enumerate(baseline_methods):
            fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'Non-Smooth {method}', line=dict(color=cols[i + len(methods)])))
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
    fig.write_image(os.path.join(log_dir, 'plots', 'legend_only.png'))



def model_rollout(net, args):

    log_dir = os.path.join(args.output_dir, args.dataset, 'predictions_long_term', net.model.name)

    net = net.to('cuda')
    net.eval()
    net.freeze()

    # get all the test video ids
    data_filepath_base = os.path.join(args.data_filepath, args.dataset)
    with open(os.path.join(data_filepath_base, 'datainfo', f'data_split_dict_{args.seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    test_vid_ids = seq_dict['test']

    pred_len = 60 if args.dataset != 'cylindrical_flow' else 200 # HARD CODED PRED LENGTH
    long_term_folder = os.path.join(log_dir, 'model_rollout')
    mkdir(long_term_folder)
    loss_dict = {}

    if 'smooth' in net.model.name:
        for p_vid_idx in tqdm(test_vid_ids):
            vid_filepath = os.path.join(data_filepath_base, str(p_vid_idx))
            total_num_frames = len(os.listdir(vid_filepath))
            suf = os.listdir(vid_filepath)[0].split('.')[-1]
            save_filepath = os.path.join(long_term_folder, str(p_vid_idx))
            mkdir(save_filepath)
            data = None
            loss_lst = []
            for start_frame_idx in range(total_num_frames - 3):
                if start_frame_idx % 2 != 0:
                    continue
                # take the initial input from ground truth data
                if start_frame_idx % pred_len == 0:
                    data = [get_data(os.path.join(vid_filepath, f'{start_frame_idx}.{suf}')),
                            get_data(os.path.join(vid_filepath, f'{start_frame_idx+1}.{suf}'))]
                    data = (torch.cat(data, 2)).unsqueeze(0)
                    # save (0', 1')
                    img = tensor_to_img(data[0, :, :, :128])
                    img.save(os.path.join(long_term_folder, f'{p_vid_idx}/' + f'{0}'+ f'.{suf}'))
                    img = tensor_to_img(data[0, :, :, 128:])
                    img.save(os.path.join(long_term_folder, f'{p_vid_idx}/' + f'{1}'+ f'.{suf}'))
                    
                # get the target
                target = [get_data(os.path.join(vid_filepath, f'{start_frame_idx+2}.{suf}')),
                          get_data(os.path.join(vid_filepath, f'{start_frame_idx+3}.{suf}'))]
                target = (torch.cat(target, 2)).unsqueeze(0)
                # feed into the model

                state, latent_gt = net.model.encoder.nsv_encoder(data.cuda())

                output, latent = net.model.decoder.nsv_decoder(state)

                # compute loss
                loss_lst.append(float(net.loss_func(output, target.cuda()).mean().cpu().detach().numpy()))

                # save (2', 3'), (4', 5'), ...
                img = tensor_to_img(output[0, :, :, :128])
                img.save(os.path.join(save_filepath, f'{start_frame_idx+2}'+ f'.{suf}'))
                img = tensor_to_img(output[0, :, :, 128:])
                img.save(os.path.join(save_filepath, f'{start_frame_idx+3}'+ f'.{suf}'))

                # the output becomes the input data in the next iteration

                data = torch.tensor(output.cpu().detach().numpy()).float()

            loss_dict[p_vid_idx] = loss_lst

        # save the test loss for all the testing videos
        with open(os.path.join(long_term_folder, 'test_loss.json'), 'w') as file:
            json.dump(loss_dict, file, indent=4)

    elif 'base' in net.model.name:
        for p_vid_idx in tqdm(test_vid_ids):
            vid_filepath = os.path.join(data_filepath_base, str(p_vid_idx))
            total_num_frames = len(os.listdir(vid_filepath))
            suf = os.listdir(vid_filepath)[0].split('.')[-1]
            data = None
            save_filepath = os.path.join(long_term_folder, str(p_vid_idx))
            mkdir(save_filepath)
            loss_lst = []
            for start_frame_idx in range(total_num_frames - 3):
                if start_frame_idx % 2 != 0:
                    continue
                # take the initial input from ground truth data
                if start_frame_idx % pred_len == 0:
                    data = [get_data(os.path.join(vid_filepath, f'{start_frame_idx}.{suf}')),
                            get_data(os.path.join(vid_filepath, f'{start_frame_idx+1}.{suf}'))]
                    data = (torch.cat(data, 2)).unsqueeze(0)
                    # save (0', 1')
                    img = tensor_to_img(data[0, :, :, :128])
                    img.save(os.path.join(long_term_folder, f'{p_vid_idx}/' + f'{0}'+ f'.{suf}'))
                    img = tensor_to_img(data[0, :, :, 128:])
                    img.save(os.path.join(long_term_folder, f'{p_vid_idx}/' + f'{1}'+ f'.{suf}'))

                # get the target
                target = [get_data(os.path.join(vid_filepath, f'{start_frame_idx+2}.{suf}')),
                          get_data(os.path.join(vid_filepath, f'{start_frame_idx+3}.{suf}'))]
                target = (torch.cat(target, 2)).unsqueeze(0)
                # feed into the model

                output, latent, state, latent_gt   = net.model(data.cuda())

                # compute loss
                loss_lst.append(float(net.loss_func(output, target.cuda()).mean().cpu().detach().numpy()))

                # save (2', 3'), (4', 5'), ...
                img = tensor_to_img(output[0, :, :, :128])
                img.save(os.path.join(save_filepath, f'{start_frame_idx+2}'+ f'.{suf}'))
                img = tensor_to_img(output[0, :, :, 128:])
                img.save(os.path.join(save_filepath, f'{start_frame_idx+3}'+ f'.{suf}'))

                # the output becomes the input data in the next iteration
                data = torch.tensor(output.cpu().detach().numpy()).float()

            loss_dict[p_vid_idx] = loss_lst

        # save the test loss for all the testing videos
        with open(os.path.join(long_term_folder, 'test_loss.json'), 'w') as file:
            json.dump(loss_dict, file, indent=4)
    
    else:
        for p_vid_idx in tqdm(test_vid_ids):
            vid_filepath = os.path.join(data_filepath_base, str(p_vid_idx))
            total_num_frames = len(os.listdir(vid_filepath))
            suf = os.listdir(vid_filepath)[0].split('.')[-1]
            data = None
            save_filepath = os.path.join(long_term_folder, str(p_vid_idx))
            mkdir(save_filepath)
            loss_lst = []
            for start_frame_idx in range(total_num_frames - 3):
                if start_frame_idx % 2 != 0:
                    continue
                # take the initial input from ground truth data
                if start_frame_idx % pred_len == 0:
                    data = [get_data(os.path.join(vid_filepath, f'{start_frame_idx}.{suf}')),
                            get_data(os.path.join(vid_filepath, f'{start_frame_idx+1}.{suf}'))]
                    data = (torch.cat(data, 2)).unsqueeze(0)

                    # save (0', 1')
                    img = tensor_to_img(data[0, :, :, :128])
                    img.save(os.path.join(long_term_folder, f'{p_vid_idx}/' + f'{0}'+ f'.{suf}'))
                    img = tensor_to_img(data[0, :, :, 128:])
                    img.save(os.path.join(long_term_folder, f'{p_vid_idx}/' + f'{1}'+ f'.{suf}'))

                # get the target
                target = [get_data(os.path.join(vid_filepath, f'{start_frame_idx+2}.{suf}')),
                          get_data(os.path.join(vid_filepath, f'{start_frame_idx+3}.{suf}'))]
                target = (torch.cat(target, 2)).unsqueeze(0)
                # feed into the model
                output, latent = net.model(data.cuda())

                # compute loss
                loss_lst.append(float(net.loss_func(output, target.cuda()).mean().cpu().detach().numpy()))
                
                # save (2', 3'), (4', 5'), ...
                img = tensor_to_img(output[0, :, :, :128])
                img.save(os.path.join(save_filepath, f'{start_frame_idx+2}'+ f'.{suf}'))
                img = tensor_to_img(output[0, :, :, 128:])
                img.save(os.path.join(save_filepath, f'{start_frame_idx+3}'+ f'.{suf}'))

                # the output becomes the input data in the next iteration
                data = torch.tensor(output.cpu().detach().numpy()).float()

            loss_dict[p_vid_idx] = loss_lst

        # save the test loss for all the testing videos
        with open(os.path.join(long_term_folder, 'test_loss.json'), 'w') as file:
            json.dump(loss_dict, file, indent=4)

    evaluate_physics_directory(args.dataset, long_term_folder, test_vid_ids)
    generate_video_directory(long_term_folder, test_vid_ids, delete_after=True)


def model_rollout_hybrid(net, args, step):

    if 'base' not in net.model.name and 'smooth' not in net.model.name:
        return False

    log_dir = os.path.join(args.output_dir, args.dataset, 'predictions_long_term', net.model.name)

    net = net.to('cuda')
    net.eval()
    net.freeze()

    # get all the test video ids
    data_filepath_base = os.path.join(args.data_filepath, args.dataset)
    with open(os.path.join(data_filepath_base, 'datainfo', f'data_split_dict_{args.seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    test_vid_ids = seq_dict['test']

    pred_len = 60 if args.dataset != 'cylindrical_flow' else 200 # HARD CODED PRED LENGTH
    long_term_folder = os.path.join(log_dir, f'hybrid_rollout_{step}')
    mkdir(long_term_folder)
    loss_dict = {}

    if 'smooth' in net.model.name:
        for p_vid_idx in tqdm(test_vid_ids):
            vid_filepath = os.path.join(data_filepath_base, str(p_vid_idx))
            total_num_frames = len(os.listdir(vid_filepath))
            suf = os.listdir(vid_filepath)[0].split('.')[-1]
            data = None
            save_filepath = os.path.join(long_term_folder, str(p_vid_idx))
            mkdir(save_filepath)
            loss_lst = []
            for start_frame_idx in range(total_num_frames - 3):
                if start_frame_idx % 2 != 0:
                    continue
                # take the initial input from ground truth data
                if start_frame_idx % pred_len == 0:
                    data = [get_data(os.path.join(vid_filepath, f'{start_frame_idx}.{suf}')),
                            get_data(os.path.join(vid_filepath, f'{start_frame_idx+1}.{suf}'))]
                    data = (torch.cat(data, 2)).unsqueeze(0)
                    # save (0', 1')
                    img = tensor_to_img(data[0, :, :, :128])
                    img.save(os.path.join(long_term_folder, f'{p_vid_idx}/' + f'{0}'+ f'.{suf}'))
                    img = tensor_to_img(data[0, :, :, 128:])
                    img.save(os.path.join(long_term_folder, f'{p_vid_idx}/' + f'{1}'+ f'.{suf}'))
                # get the target
                target = [get_data(os.path.join(vid_filepath, f'{start_frame_idx+2}.{suf}')),
                        get_data(os.path.join(vid_filepath, f'{start_frame_idx+3}.{suf}'))]
                target = (torch.cat(target, 2)).unsqueeze(0)
                # feed into the model
                if (start_frame_idx + 2) % (2 * step + 2) == 0:
                    state, latent_gt = net.model.encoder.nsv_encoder(data.cuda())
                    #if net.model.decoder.nsv_decoder.angular:
                        #print("angualr")
                     #   state = net.model.decoder.cosSin(state)
                    output, latent = net.model.decoder.nsv_decoder(state)
                else:
                    output = net.model.decoder.nsv_decoder.latent_decoder(net.model.encoder.nsv_encoder.latent_encoder(data.cuda()))

                # compute loss
                loss_lst.append(float(net.loss_func(output, target.cuda()).mean().cpu().detach().numpy())) 
                                      
                # save (2', 3'), (4', 5'), ...
                img = tensor_to_img(output[0, :, :, :128])
                img.save(os.path.join(save_filepath, f'{start_frame_idx+2}'+ f'.{suf}'))
                img = tensor_to_img(output[0, :, :, 128:])
                img.save(os.path.join(save_filepath, f'{start_frame_idx+3}'+ f'.{suf}'))

                # the output becomes the input data in the next iteration
                data = torch.tensor(output.cpu().detach().numpy()).float()

            loss_dict[p_vid_idx] = loss_lst

        # save the test loss for all the testing videos
        with open(os.path.join(long_term_folder, 'test_loss.json'), 'w') as file:
            json.dump(loss_dict, file, indent=4)
    
    else:
        for p_vid_idx in tqdm(test_vid_ids):
            vid_filepath = os.path.join(data_filepath_base, str(p_vid_idx))
            total_num_frames = len(os.listdir(vid_filepath))
            suf = os.listdir(vid_filepath)[0].split('.')[-1]
            data = None
            save_filepath = os.path.join(long_term_folder, str(p_vid_idx))
            mkdir(save_filepath)
            loss_lst = []
            for start_frame_idx in range(total_num_frames - 3):
                if start_frame_idx % 2 != 0:
                    continue
                # take the initial input from ground truth data
                if start_frame_idx % pred_len == 0:
                    data = [get_data(os.path.join(vid_filepath, f'{start_frame_idx}.{suf}')),
                            get_data(os.path.join(vid_filepath, f'{start_frame_idx+1}.{suf}'))]
                    data = (torch.cat(data, 2)).unsqueeze(0)

                    # save (0', 1')
                    img = tensor_to_img(data[0, :, :, :128])
                    img.save(os.path.join(long_term_folder, f'{p_vid_idx}/' + f'{0}'+ f'.{suf}'))
                    img = tensor_to_img(data[0, :, :, 128:])
                    img.save(os.path.join(long_term_folder, f'{p_vid_idx}/' + f'{1}'+ f'.{suf}'))
                # get the target
                target = [get_data(os.path.join(vid_filepath, f'{start_frame_idx+2}.{suf}')),
                        get_data(os.path.join(vid_filepath, f'{start_frame_idx+3}.{suf}'))]
                target = (torch.cat(target, 2)).unsqueeze(0)
                # feed into the model
                if (start_frame_idx + 2) % (2 * step + 2) == 0:
                    output, latent, state, latent_gt  = net.model(data.cuda())
                else:
                    output = net.model.decoder.latent_decoder(net.model.encoder.latent_encoder(data.cuda()))

                # compute loss
                loss_lst.append(float(net.loss_func(output, target.cuda()).mean().cpu().detach().numpy()))

                # save (2', 3'), (4', 5'), ...
                img = tensor_to_img(output[0, :, :, :128])
                img.save(os.path.join(save_filepath, f'{start_frame_idx+2}'+ f'.{suf}'))
                img = tensor_to_img(output[0, :, :, 128:])
                img.save(os.path.join(save_filepath, f'{start_frame_idx+3}'+ f'.{suf}'))
                
                # the output becomes the input data in the next iteration
                data = torch.tensor(output.cpu().detach().numpy()).float()

            loss_dict[p_vid_idx] = loss_lst

        # save the test loss for all the testing videos
        with open(os.path.join(long_term_folder, 'test_loss.json'), 'w') as file:
            json.dump(loss_dict, file, indent=4)
    
    
    evaluate_physics_directory(args.dataset, long_term_folder, test_vid_ids)
    generate_video_directory(long_term_folder, test_vid_ids, delete_after=True)

    return True

def get_data(filepath):
    data = Image.open(filepath)
    data = data.resize((128, 128))
    data = np.array(data)
    data = torch.tensor(data / 255.0)
    data = data.permute(2, 0, 1).float()
    return data

# out_tensor: 3 x 128 x 128 -> 128 x 128 x 3
def tensor_to_img(out_tensor):
    return transforms.ToPILImage()(out_tensor).convert("RGB")
