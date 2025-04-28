from utils.misc import *
import numpy as np
import argparse
from munch import munchify
from utils.analysis import Physics_Evaluator
from utils.show import *


methods = ['model_rollout', 'hybrid_rollout_3']

def remove_outlier(x, percentile=98):
    if x.size == 0:
        return x
    else:
        thresh = np.percentile(x, percentile)
        return x[x <= thresh]


def summarize_nsvf(args, reject_threshold = 0.5):

    save_path = os.path.join(args.output_dir, args.dataset, 'summary', 'nsvf_predictions_long_term')
    mkdir(save_path)

    pred_len = 60 if args.dataset != 'cylindrical_flow' else 200
    dt = 1/60 if args.dataset != 'cylindrical_flow' else .02
    t = np.linspace(0,dt*pred_len, pred_len).tolist()

    nsvf_data_path = os.path.join(args.output_dir, args.dataset, 'tasks')
    data_path = os.path.join(args.output_dir, args.dataset, 'predictions_long_term')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]
    
    smooth_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'regress-smooth-filtered.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        smooth_models.append(name)
    
    
    refine_models = [f'base_{seed}' for seed in seeds]
    latent64_models = [f'encoder-decoder-64_{seed}' for seed in seeds]
    latent8192_models = [f'encoder-decoder_{seed}' for seed in seeds]
    
    smooth_losses = {smooth_model: np.load(os.path.join(nsvf_data_path, smooth_model, 'mlp_predictions/losses.npy' ), allow_pickle=True).item() for smooth_model in smooth_models}
    refine_losses = {refine_model: np.load(os.path.join(data_path, refine_model, 'losses.npy' ), allow_pickle=True).item() for refine_model in refine_models}
    latent64_losses = {latent64_model: np.load(os.path.join(data_path, latent64_model, 'losses.npy' ), allow_pickle=True).item() for latent64_model in latent64_models}
    latent8192_losses = {latent8192_model: np.load(os.path.join(data_path, latent8192_model, 'losses.npy' ), allow_pickle=True).item() for latent8192_model in latent8192_models}

    evaluator = Physics_Evaluator(args.dataset)
    phys_vars_list = evaluator.get_phys_vars(False)

    figs = [go.Figure() for _ in range(len(phys_vars_list) + 1)]

    all_smooth_losses = {var: [] for var in phys_vars_list}
    all_refine_losses = {method: {var: [] for var in phys_vars_list} for method in methods}
    all_latent64_losses = {var: [] for var in phys_vars_list}
    all_latent8192_losses = {var: [] for var in phys_vars_list}

    smooth_reject = []
    smooth_reject_data = []

    refine_reject = {method: [] for method in methods}
    refine_reject_data = {method: [] for method in methods}
    refine_reject_ratio = {}

    latent64_reject = []
    latent64_reject_data = []
    latent8192_reject = []
    latent8192_reject_data = []


    for smooth_model in smooth_models:
        smooth_reject.append(smooth_losses[smooth_model]['reject'])
        smooth_reject_data.append(smooth_losses[smooth_model]['reject_data'])

        for var in phys_vars_list:
            all_smooth_losses[var].append(smooth_losses[smooth_model][var])
    
    smooth_reject = np.concatenate(smooth_reject)
    smooth_reject_data = np.concatenate(smooth_reject_data)
    for var in phys_vars_list:
        all_smooth_losses[var] = np.concatenate(all_smooth_losses[var])

    smooth_reject_ratio = scale_reject_ratio(pred_len, smooth_reject, smooth_reject_data)
    figs[0].add_trace(go.Scatter(x=t, y=smooth_reject_ratio, line=dict(color=cols[0], width=7), name='NSVF Integration'))


    for i, method in enumerate(methods):

        for refine_model in refine_models:
            refine_reject[method].append(refine_losses[refine_model][method]['reject'])
            refine_reject_data[method].append(refine_losses[refine_model][method]['reject_data'])
        
            for var in phys_vars_list:
                all_refine_losses[method][var].append(refine_losses[refine_model][method][var])
        
        refine_reject[method] = np.concatenate(refine_reject[method])
        refine_reject_data[method] = np.concatenate(refine_reject_data[method])
        for var in phys_vars_list:
            all_refine_losses[method][var] = np.concatenate(all_refine_losses[method][var])
        
        refine_reject_ratio[method] = scale_reject_ratio(pred_len, refine_reject[method], refine_reject_data[method])
        figs[0].add_trace(go.Scatter(x=t, y=refine_reject_ratio[method], line=dict(color=cols[2+i], width=7), name=f'Non Smooth {method}'))

    
    for latent64_model in latent64_models:
        latent64_reject.append(latent64_losses[latent64_model]['model_rollout']['reject'])
        latent64_reject_data.append(latent64_losses[latent64_model]['model_rollout']['reject_data'])

        for var in phys_vars_list:
            all_latent64_losses[var].append(latent64_losses[latent64_model]['model_rollout'][var])
        
    latent64_reject = np.concatenate(latent64_reject)
    latent64_reject_data = np.concatenate(latent64_reject_data)
    for var in phys_vars_list:
        all_latent64_losses[var] = np.concatenate(all_latent64_losses[var])
    
    latent64_reject_ratio = scale_reject_ratio(pred_len, latent64_reject, latent64_reject_data)
    figs[0].add_trace(go.Scatter(x=t, y=latent64_reject_ratio, line=dict(color=cols[4], width=7), name='Dim-64'))
    
    
    for latent8192_model in latent8192_models:
        latent8192_reject.append(latent8192_losses[latent8192_model]['model_rollout']['reject'])
        latent8192_reject_data.append(latent8192_losses[latent8192_model]['model_rollout']['reject_data'])

        for var in phys_vars_list:
            all_latent8192_losses[var].append(latent8192_losses[latent8192_model]['model_rollout'][var])
        
    latent8192_reject = np.concatenate(latent8192_reject)
    latent8192_reject_data = np.concatenate(latent8192_reject_data)
    for var in phys_vars_list:
        all_latent8192_losses[var] = np.concatenate(all_latent8192_losses[var])
    
    latent8192_reject_ratio = scale_reject_ratio(pred_len, latent8192_reject, latent8192_reject_data)
    figs[0].add_trace(go.Scatter(x=t, y=latent8192_reject_ratio, line=dict(color=cols[5], width=7), name='Dim-8192'))
    
    update_figure(figs[0], True)
    # figs[0].update_layout( xaxis=dict(title='<b>Time (s)</b>', range=[0, dt*(pred_len+1)], nticks=3),
    #                         yaxis=dict(title='<b>Reject Ratio</b>', range=[-.01,1.05], tick0=0, dtick=0.5), 
    #                         showlegend=False)
    figs[0].update_layout( xaxis=dict( range=[0, dt*(pred_len+1)], nticks=3),
                            yaxis=dict(range=[-.01,1.05], tick0=0, dtick=0.5), 
                            showlegend=False)

    smooth_phys_error_mean = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}
    smooth_phys_error_sem = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}
    
    refine_phys_error_mean = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}
    refine_phys_error_sem = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}

    latent64_phys_error_mean = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}
    latent64_phys_error_sem = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}

    latent8192_phys_error_mean = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}
    latent8192_phys_error_sem = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}

    for i, var in enumerate(phys_vars_list):
        for j, method in enumerate(methods):
            refine_max = pred_len
            for p in range(pred_len):
            
                refine_error_p = all_refine_losses[method][var][:, p]
                refine_error_p = refine_error_p[~np.isnan(refine_error_p)]
                refine_error_p = remove_outlier(refine_error_p)
                if refine_error_p.size / all_refine_losses[method][var][:, p].size > reject_threshold:
                    refine_phys_error_mean[var][method][p] = np.mean(refine_error_p)
                    refine_phys_error_sem[var][method][p] = stats.sem(refine_error_p)
                else:
                    refine_phys_error_mean[var][method][p] = np.nan
                    refine_phys_error_sem[var][method][p] = np.nan
                    if refine_max == pred_len:
                        refine_max = p

            figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=refine_phys_error_mean[var][method], line=dict(color=cols[2+j], width=7), name=f'Non Smooth {method}'))
            error_upper = (refine_phys_error_mean[var][method] + refine_phys_error_sem[var][method]).tolist()
            error_lower = (refine_phys_error_mean[var][method] - refine_phys_error_sem[var][method]).tolist()
            figs[i+1].add_trace(go.Scatter(x=t[:refine_max] + t[:refine_max][::-1], y=error_upper[:refine_max] + error_lower[:refine_max][::-1], fill='toself', fillcolor=cols[2+j], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
            
    
        smooth_max = pred_len
        latent64_max = pred_len
        latent8192_max = pred_len
        for p in range(pred_len):

            smooth_error_p = all_smooth_losses[var][:, p]
            smooth_error_p = smooth_error_p[~np.isnan(smooth_error_p)]
            smooth_error_p = remove_outlier(smooth_error_p)
            if smooth_error_p.size / all_smooth_losses[var][:, p].size > reject_threshold:
                smooth_phys_error_mean[var][p] = np.mean(smooth_error_p)
                smooth_phys_error_sem[var][p] = stats.sem(smooth_error_p)
            else:
                smooth_phys_error_mean[var][p] = np.nan
                smooth_phys_error_sem[var][p] = np.nan
                if smooth_max == pred_len:
                    smooth_max = p

            latent64_error_p = all_latent64_losses[var][:, p]
            latent64_error_p = latent64_error_p[~np.isnan(latent64_error_p)]
            latent64_error_p = remove_outlier(latent64_error_p)
            if latent64_error_p.size / all_latent64_losses[var][:, p].size > reject_threshold:
                latent64_phys_error_mean[var][p] = np.mean(latent64_error_p)
                latent64_phys_error_sem[var][p] = stats.sem(latent64_error_p)
            else:
                latent64_phys_error_mean[var][p] = np.nan
                latent64_phys_error_sem[var][p] = np.nan
                if latent64_max == pred_len:
                    latent64_max = p
        
            latent8192_error_p = all_latent8192_losses[var][:, p]
            latent8192_error_p = latent8192_error_p[~np.isnan(latent8192_error_p)]
            latent8192_error_p = remove_outlier(latent8192_error_p)
            if latent8192_error_p.size / all_latent8192_losses[var][:, p].size > reject_threshold:
                latent8192_phys_error_mean[var][p] = np.mean(latent8192_error_p)
                latent8192_phys_error_sem[var][p] = stats.sem(latent8192_error_p)
            else:
                latent8192_phys_error_mean[var][p] = np.nan
                latent8192_phys_error_sem[var][p] = np.nan
                if latent8192_max == pred_len:
                    latent8192_max = p
        
        figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=smooth_phys_error_mean[var], line=dict(color=cols[0], width=7), name='NSVF Integration'))
        error_upper = (smooth_phys_error_mean[var] + smooth_phys_error_sem[var]).tolist()
        error_lower = (smooth_phys_error_mean[var] - smooth_phys_error_sem[var]).tolist()
        figs[i+1].add_trace(go.Scatter(x=t[:smooth_max] + t[:smooth_max][::-1], y=error_upper[:smooth_max] + error_lower[:smooth_max][::-1], fill='toself', fillcolor=cols[j], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        
        
        figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=latent64_phys_error_mean[var], line=dict(color=cols[4], width=7), name='Dim-64'))
        error_upper = (latent64_phys_error_mean[var] + latent64_phys_error_sem[var]).tolist()
        error_lower = (latent64_phys_error_mean[var] - latent64_phys_error_sem[var]).tolist()
        figs[i+1].add_trace(go.Scatter(x=t[:latent64_max] + t[:latent64_max][::-1], y=error_upper[:latent64_max] + error_lower[:latent64_max][::-1], fill='toself', fillcolor=cols[4], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        
        figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=latent8192_phys_error_mean[var], line=dict(color=cols[5], width=7), name='Dim-8192'))
        error_upper = (latent8192_phys_error_mean[var] + latent8192_phys_error_sem[var]).tolist()
        error_lower = (latent8192_phys_error_mean[var] - latent8192_phys_error_sem[var]).tolist()
        figs[i+1].add_trace(go.Scatter(x=t[:latent8192_max] + t[:latent8192_max][::-1], y=error_upper[:latent8192_max] + error_lower[:latent8192_max][::-1], fill='toself', fillcolor=cols[5], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        

        update_figure(figs[i+1], True)
        # figs[i+1].update_layout(#title=f'{var} error (L1)', 
        #                     xaxis=dict(title='<b>Time (s)</b>', range=[0, dt*(pred_len+1)], nticks=3),
        #                     yaxis=dict(title="<b>L1 error</b>", tick0=0, nticks=6), 
        #                     showlegend=False)
        figs[i+1].update_layout(#title=f'{var} error (L1)', 
                            xaxis=dict(range=[0, dt*(pred_len+1)], nticks=3),
                            yaxis=dict(tick0=0, nticks=6), 
                            showlegend=False)
    
    phys_vars_list = evaluator.get_phys_vars(True)
    for i, f in enumerate(figs):
        f.write_image(os.path.join(save_path, f'{phys_vars_list[i]}.png'), scale=4)
    
    fig = go.Figure()
    for i, method in enumerate(methods):
        fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'Non Smooth {method}', line=dict(color=cols[i + 2])))
    fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name='NSVF Integration', line=dict(color=cols[0])))
    fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name='Dim-64', line=dict(color=cols[4])))
    fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name='Dim-8192', line=dict(color=cols[5])))
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
    fig.write_image(os.path.join(save_path,'legend_only.png'), scale=4)


def summarize_nsv(args, reject_threshold=0.5):

    save_path = os.path.join(args.output_dir, args.dataset, 'summary', 'nsv_predictions_long_term')
    mkdir(save_path)

    pred_len = 60 if args.dataset != 'cylindrical_flow' else 200
    dt = 1/60 if args.dataset != 'cylindrical_flow' else .02
    t = np.linspace(0,dt*pred_len, pred_len).tolist()

    data_path = os.path.join(args.output_dir, args.dataset, 'predictions_long_term')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]
    
    smooth_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'smooth.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        smooth_models.append(name)
    
    refine_models = [f'base_{seed}' for seed in seeds]
    latent64_models = [f'encoder-decoder-64_{seed}' for seed in seeds]
    latent8192_models = [f'encoder-decoder_{seed}' for seed in seeds]
    
    smooth_losses = {smooth_model: np.load(os.path.join(data_path, smooth_model, 'losses.npy' ), allow_pickle=True).item() for smooth_model in smooth_models}
    refine_losses = {refine_model: np.load(os.path.join(data_path, refine_model, 'losses.npy' ), allow_pickle=True).item() for refine_model in refine_models}
    latent64_losses = {latent64_model: np.load(os.path.join(data_path, latent64_model, 'losses.npy' ), allow_pickle=True).item() for latent64_model in latent64_models}
    latent8192_losses = {latent8192_model: np.load(os.path.join(data_path, latent8192_model, 'losses.npy' ), allow_pickle=True).item() for latent8192_model in latent8192_models}

    evaluator = Physics_Evaluator(args.dataset)
    phys_vars_list = evaluator.get_phys_vars(False)

    figs = [go.Figure() for _ in range(len(phys_vars_list) + 1)]

    all_smooth_losses = {method: {var: [] for var in phys_vars_list} for method in methods}
    all_refine_losses = {method: {var: [] for var in phys_vars_list} for method in methods}
    all_latent64_losses = {var: [] for var in phys_vars_list}
    all_latent8192_losses = {var: [] for var in phys_vars_list}

    smooth_reject = {method: [] for method in methods}
    smooth_reject_data = {method: [] for method in methods}
    smooth_reject_ratio = {}
    refine_reject = {method: [] for method in methods}
    refine_reject_data = {method: [] for method in methods}
    refine_reject_ratio = {}

    latent64_reject = []
    latent64_reject_data = []
    latent8192_reject = []
    latent8192_reject_data = []


    for i, method in enumerate(methods):

        for smooth_model in smooth_models:
            smooth_reject[method].append(smooth_losses[smooth_model][method]['reject'])
            smooth_reject_data[method].append(smooth_losses[smooth_model][method]['reject_data'])

            for var in phys_vars_list:
                all_smooth_losses[method][var].append(smooth_losses[smooth_model][method][var])
        
        smooth_reject[method] = np.concatenate(smooth_reject[method])
        smooth_reject_data[method] = np.concatenate(smooth_reject_data[method])
        for var in phys_vars_list:
                all_smooth_losses[method][var] = np.concatenate(all_smooth_losses[method][var])

        smooth_reject_ratio[method] = scale_reject_ratio(pred_len, smooth_reject[method], smooth_reject_data[method])
        figs[0].add_trace(go.Scatter(x=t, y=smooth_reject_ratio[method], line=dict(color=cols[i], width=7), name=f'Smooth {method}'))
    
        
        for refine_model in refine_models:
            refine_reject[method].append(refine_losses[refine_model][method]['reject'])
            refine_reject_data[method].append(refine_losses[refine_model][method]['reject_data'])
        
            for var in phys_vars_list:
                all_refine_losses[method][var].append(refine_losses[refine_model][method][var])
        
        refine_reject[method] = np.concatenate(refine_reject[method])
        refine_reject_data[method] = np.concatenate(refine_reject_data[method])
        for var in phys_vars_list:
            all_refine_losses[method][var] = np.concatenate(all_refine_losses[method][var])
        
        refine_reject_ratio[method] = scale_reject_ratio(pred_len, refine_reject[method], refine_reject_data[method])
        figs[0].add_trace(go.Scatter(x=t, y=refine_reject_ratio[method], line=dict(color=cols[2+i], width=7), name=f'Non Smooth {method}'))
    
    
    for latent64_model in latent64_models:
        latent64_reject.append(latent64_losses[latent64_model]['model_rollout']['reject'])
        latent64_reject_data.append(latent64_losses[latent64_model]['model_rollout']['reject_data'])

        for var in phys_vars_list:
            all_latent64_losses[var].append(latent64_losses[latent64_model]['model_rollout'][var])
        
    latent64_reject = np.concatenate(latent64_reject)
    latent64_reject_data = np.concatenate(latent64_reject_data)
    for var in phys_vars_list:
        all_latent64_losses[var] = np.concatenate(all_latent64_losses[var])
    
    latent64_reject_ratio = scale_reject_ratio(pred_len, latent64_reject, latent64_reject_data)
    figs[0].add_trace(go.Scatter(x=t, y=latent64_reject_ratio, line=dict(color=cols[4], width=7), name='Dim-64'))
    
    
    for latent8192_model in latent8192_models:
        latent8192_reject.append(latent8192_losses[latent8192_model]['model_rollout']['reject'])
        latent8192_reject_data.append(latent8192_losses[latent8192_model]['model_rollout']['reject_data'])

        for var in phys_vars_list:
            all_latent8192_losses[var].append(latent8192_losses[latent8192_model]['model_rollout'][var])
        
    latent8192_reject = np.concatenate(latent8192_reject)
    latent8192_reject_data = np.concatenate(latent8192_reject_data)
    for var in phys_vars_list:
        all_latent8192_losses[var] = np.concatenate(all_latent8192_losses[var])
    
    latent8192_reject_ratio = scale_reject_ratio(pred_len, latent8192_reject, latent8192_reject_data)
    figs[0].add_trace(go.Scatter(x=t, y=latent8192_reject_ratio, line=dict(color=cols[5], width=7), name='Dim-8192'))
    
    update_figure(figs[0], True)
    # figs[0].update_layout( xaxis=dict(title='<b>Time (s)</b>', range=[0, dt*(pred_len+1)], nticks=3),
    #                         yaxis=dict(title='<b>Reject Ratio</b>', range=[-.01,1.05], tick0=0, dtick=0.5), 
    #                         showlegend=False)
    figs[0].update_layout( xaxis=dict(range=[0, dt*(pred_len+1)], nticks=3),
                            yaxis=dict(range=[-.01,1.05], tick0=0, dtick=0.5), 
                            showlegend=False)

    smooth_phys_error_mean = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}
    smooth_phys_error_sem = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}
    
    refine_phys_error_mean = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}
    refine_phys_error_sem = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}

    latent64_phys_error_mean = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}
    latent64_phys_error_sem = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}

    latent8192_phys_error_mean = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}
    latent8192_phys_error_sem = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}


    for i, var in enumerate(phys_vars_list):
        for j, method in enumerate(methods):
            smooth_max = pred_len
            refine_max = pred_len
            for p in range(pred_len):
                smooth_error_p = all_smooth_losses[method][var][:, p]
                smooth_error_p = smooth_error_p[~np.isnan(smooth_error_p)]
                smooth_error_p = remove_outlier(smooth_error_p)
                if smooth_error_p.size / all_smooth_losses[method][var][:, p].size > reject_threshold:
                    smooth_phys_error_mean[var][method][p] = np.mean(smooth_error_p)
                    smooth_phys_error_sem[var][method][p] = stats.sem(smooth_error_p)
                else:
                    smooth_phys_error_mean[var][method][p] = np.nan
                    smooth_phys_error_sem[var][method][p] = np.nan
                    if smooth_max == pred_len:
                        smooth_max = p
            
                refine_error_p = all_refine_losses[method][var][:, p]
                refine_error_p = refine_error_p[~np.isnan(refine_error_p)]
                refine_error_p = remove_outlier(refine_error_p)
                if refine_error_p.size / all_refine_losses[method][var][:, p].size > reject_threshold:
                    refine_phys_error_mean[var][method][p] = np.mean(refine_error_p)
                    refine_phys_error_sem[var][method][p] = stats.sem(refine_error_p)
                else:
                    refine_phys_error_mean[var][method][p] = np.nan
                    refine_phys_error_sem[var][method][p] = np.nan
                    if refine_max == pred_len:
                        refine_max = p


            figs[i+1].add_trace(go.Scatter(x=t, y=smooth_phys_error_mean[var][method], mode='lines', line=dict(color=cols[j], width=7), name=f'Smooth {method}'))
            error_upper = (smooth_phys_error_mean[var][method] + smooth_phys_error_sem[var][method]).tolist()
            error_lower = (smooth_phys_error_mean[var][method] - smooth_phys_error_sem[var][method]).tolist()
            figs[i+1].add_trace(go.Scatter(x=t[:smooth_max] + t[:smooth_max][::-1], y=error_upper[:smooth_max] + error_lower[:smooth_max][::-1], fill='toself', fillcolor=cols[j], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
            
            
            figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=refine_phys_error_mean[var][method], line=dict(color=cols[2+j], width=7), name=f'Non Smooth {method}'))
            error_upper = (refine_phys_error_mean[var][method] + refine_phys_error_sem[var][method]).tolist()
            error_lower = (refine_phys_error_mean[var][method] - refine_phys_error_sem[var][method]).tolist()
            figs[i+1].add_trace(go.Scatter(x=t[:refine_max] + t[:refine_max][::-1], y=error_upper[:refine_max] + error_lower[:refine_max][::-1], fill='toself', fillcolor=cols[2+j], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
            
    
        latent64_max = pred_len
        latent8192_max = pred_len
        for p in range(pred_len):
            latent64_error_p = all_latent64_losses[var][:, p]
            latent64_error_p = latent64_error_p[~np.isnan(latent64_error_p)]
            latent64_error_p = remove_outlier(latent64_error_p)
            if latent64_error_p.size / all_latent64_losses[var][:, p].size > reject_threshold:
                latent64_phys_error_mean[var][p] = np.mean(latent64_error_p)
                latent64_phys_error_sem[var][p] = stats.sem(latent64_error_p)
            else:
                latent64_phys_error_mean[var][p] = np.nan
                latent64_phys_error_sem[var][p] = np.nan
                if latent64_max == pred_len:
                    latent64_max = p
        
            latent8192_error_p = all_latent8192_losses[var][:, p]
            latent8192_error_p = latent8192_error_p[~np.isnan(latent8192_error_p)]
            latent8192_error_p = remove_outlier(latent8192_error_p)
            if latent8192_error_p.size / all_latent8192_losses[var][:, p].size > reject_threshold:
                latent8192_phys_error_mean[var][p] = np.mean(latent8192_error_p)
                latent8192_phys_error_sem[var][p] = stats.sem(latent8192_error_p)
            else:
                latent8192_phys_error_mean[var][p] = np.nan
                latent8192_phys_error_sem[var][p] = np.nan
                if latent8192_max == pred_len:
                    latent8192_max = p
        
        figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=latent64_phys_error_mean[var], line=dict(color=cols[4], width=7), name='Dim-64'))
        error_upper = (latent64_phys_error_mean[var] + latent64_phys_error_sem[var]).tolist()
        error_lower = (latent64_phys_error_mean[var] - latent64_phys_error_sem[var]).tolist()
        figs[i+1].add_trace(go.Scatter(x=t[:latent64_max] + t[:latent64_max][::-1], y=error_upper[:latent64_max] + error_lower[:latent64_max][::-1], fill='toself', fillcolor=cols[4], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        
        figs[i+1].add_trace(go.Scatter(x=np.linspace(0,dt*pred_len, pred_len), y=latent8192_phys_error_mean[var], line=dict(color=cols[5], width=7), name='Dim-8192'))
        error_upper = (latent8192_phys_error_mean[var] + latent8192_phys_error_sem[var]).tolist()
        error_lower = (latent8192_phys_error_mean[var] - latent8192_phys_error_sem[var]).tolist()
        figs[i+1].add_trace(go.Scatter(x=t[:latent8192_max] + t[:latent8192_max][::-1], y=error_upper[:latent8192_max] + error_lower[:latent8192_max][::-1], fill='toself', fillcolor=cols[5], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        
    

        update_figure(figs[i+1], True)
        # figs[i+1].update_layout(#title=f'{var} error (L1)', 
        #                     xaxis=dict(title='<b>Time (s)</b>', range=[0, dt*(pred_len+1)], nticks=3),
        #                     yaxis=dict(title="<b>L1 error</b>", tick0=0, nticks=6), 
        #                     showlegend=False)
        figs[i+1].update_layout(#title=f'{var} error (L1)', 
                            xaxis=dict(range=[0, dt*(pred_len+1)], nticks=3),
                            yaxis=dict(tick0=0, nticks=6), 
                            showlegend=False)
    
    phys_vars_list = evaluator.get_phys_vars(True)
    for i, f in enumerate(figs):
        f.write_image(os.path.join(save_path, f'{phys_vars_list[i]}.png'), scale=4)
    
    fig = go.Figure()
    for i, method in enumerate(methods):
        fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'Smooth {method}', line=dict(color=cols[i])))
        fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'Non Smooth {method}', line=dict(color=cols[i + 2])))
    fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name='Dim-64', line=dict(color=cols[4])))
    fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name='Dim-8192', line=dict(color=cols[5])))
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
    fig.write_image(os.path.join(save_path,'legend_only.png'), scale=4)



def summarize_noAnnealing(args, reject_threshold=0.5):

    save_path = os.path.join(args.output_dir, args.dataset, 'summary', 'noAnnealing')
    mkdir(save_path)

    pred_len = 60 if args.dataset != 'cylindrical_flow' else 200
    dt = 1/60 if args.dataset != 'cylindrical_flow' else .02
    t = np.linspace(0,dt*pred_len, pred_len).tolist()

    data_path = os.path.join(args.output_dir, args.dataset, 'predictions_long_term')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]
    
    smooth_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'smooth.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        smooth_models.append(name)
    
    noAnnealing_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'smooth-noAnnealing.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        noAnnealing_models.append(name)
    
    
    
    smooth_losses = {smooth_model: np.load(os.path.join(data_path, smooth_model, 'losses.npy' ), allow_pickle=True).item() for smooth_model in smooth_models}
    noAnnealing_losses = {noAnnealing_model: np.load(os.path.join(data_path, noAnnealing_model, 'losses.npy' ), allow_pickle=True).item() for noAnnealing_model in noAnnealing_models}
    
    evaluator = Physics_Evaluator(args.dataset)
    phys_vars_list = evaluator.get_phys_vars(False)

    figs = [go.Figure() for _ in range(len(phys_vars_list) + 1)]

    all_smooth_losses = {method: {var: [] for var in phys_vars_list} for method in methods}
    all_noAnnealing_losses = {method: {var: [] for var in phys_vars_list} for method in methods}
    
    smooth_reject = {method: [] for method in methods}
    smooth_reject_data = {method: [] for method in methods}
    smooth_reject_ratio = {}

    noAnnealing_reject = {method: [] for method in methods}
    noAnnealing_reject_data = {method: [] for method in methods}
    noAnnealing_reject_ratio = {}


    for i, method in enumerate(methods):

        for smooth_model in smooth_models:
            smooth_reject[method].append(smooth_losses[smooth_model][method]['reject'])
            smooth_reject_data[method].append(smooth_losses[smooth_model][method]['reject_data'])

            for var in phys_vars_list:
                all_smooth_losses[method][var].append(smooth_losses[smooth_model][method][var])
        
        smooth_reject[method] = np.concatenate(smooth_reject[method])
        smooth_reject_data[method] = np.concatenate(smooth_reject_data[method])
        for var in phys_vars_list:
                all_smooth_losses[method][var] = np.concatenate(all_smooth_losses[method][var])

        smooth_reject_ratio[method] = scale_reject_ratio(pred_len, smooth_reject[method], smooth_reject_data[method])
        figs[0].add_trace(go.Scatter(x=t, y=smooth_reject_ratio[method], line=dict(color=cols[i], width=7), name=f'Annealing {method}'))

        for noAnnealing_model in noAnnealing_models:
            noAnnealing_reject[method].append(noAnnealing_losses[noAnnealing_model][method]['reject'])
            noAnnealing_reject_data[method].append(noAnnealing_losses[noAnnealing_model][method]['reject_data'])

            for var in phys_vars_list:
                all_noAnnealing_losses[method][var].append(noAnnealing_losses[noAnnealing_model][method][var])
        
        noAnnealing_reject[method] = np.concatenate(noAnnealing_reject[method])
        noAnnealing_reject_data[method] = np.concatenate(noAnnealing_reject_data[method])
        for var in phys_vars_list:
                all_noAnnealing_losses[method][var] = np.concatenate(all_noAnnealing_losses[method][var])

        noAnnealing_reject_ratio[method] = scale_reject_ratio(pred_len, noAnnealing_reject[method], noAnnealing_reject_data[method])
        figs[0].add_trace(go.Scatter(x=t, y=noAnnealing_reject_ratio[method], line=dict(color=cols[i+2], width=7), name=f'No Annealing {method}'))
    
    
        
    update_figure(figs[0], True)
    # figs[0].update_layout( xaxis=dict(title='<b>Time (s)</b>', range=[0, dt*(pred_len+1)], nticks=3),
    #                         yaxis=dict(title='<b>Reject Ratio</b>', range=[-.01,1.05], tick0=0, dtick=0.5), 
    #                         showlegend=False)
    figs[0].update_layout( xaxis=dict(range=[0, dt*(pred_len+1)], nticks=3),
                            yaxis=dict(range=[-.01,1.05], tick0=0, dtick=0.5), 
                            showlegend=False)

    smooth_phys_error_mean = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}
    smooth_phys_error_sem = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}

    noAnnealing_phys_error_mean = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}
    noAnnealing_phys_error_sem = {p_var: {method: np.zeros(pred_len) for method in methods} for p_var in phys_vars_list}

    for i, var in enumerate(phys_vars_list):
        for j, method in enumerate(methods):
            smooth_max = pred_len
            
            for p in range(pred_len):
                smooth_error_p = all_smooth_losses[method][var][:, p]
                smooth_error_p = smooth_error_p[~np.isnan(smooth_error_p)]
                smooth_error_p = remove_outlier(smooth_error_p)
                if smooth_error_p.size / all_smooth_losses[method][var][:, p].size > reject_threshold:
                    smooth_phys_error_mean[var][method][p] = np.mean(smooth_error_p)
                    smooth_phys_error_sem[var][method][p] = stats.sem(smooth_error_p)
                else:
                    smooth_phys_error_mean[var][method][p] = np.nan
                    smooth_phys_error_sem[var][method][p] = np.nan
                    if smooth_max == pred_len:
                        smooth_max = p


            figs[i+1].add_trace(go.Scatter(x=t, y=smooth_phys_error_mean[var][method], mode='lines', line=dict(color=cols[j], width=7), name=f'Annealing {method}'))
            error_upper = (smooth_phys_error_mean[var][method] + smooth_phys_error_sem[var][method]).tolist()
            error_lower = (smooth_phys_error_mean[var][method] - smooth_phys_error_sem[var][method]).tolist()
            figs[i+1].add_trace(go.Scatter(x=t[:smooth_max] + t[:smooth_max][::-1], y=error_upper[:smooth_max] + error_lower[:smooth_max][::-1], fill='toself', fillcolor=cols[j], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
            
            noAnnealing_max = pred_len
            
            for p in range(pred_len):
                noAnnealing_error_p = all_noAnnealing_losses[method][var][:, p]
                noAnnealing_error_p = noAnnealing_error_p[~np.isnan(noAnnealing_error_p)]
                noAnnealing_error_p = remove_outlier(noAnnealing_error_p)
                if noAnnealing_error_p.size / all_noAnnealing_losses[method][var][:, p].size > reject_threshold:
                    noAnnealing_phys_error_mean[var][method][p] = np.mean(noAnnealing_error_p)
                    noAnnealing_phys_error_sem[var][method][p] = stats.sem(noAnnealing_error_p)
                else:
                    noAnnealing_phys_error_mean[var][method][p] = np.nan
                    noAnnealing_phys_error_sem[var][method][p] = np.nan
                    if noAnnealing_max == pred_len:
                        noAnnealing_max = p


            figs[i+1].add_trace(go.Scatter(x=t, y=noAnnealing_phys_error_mean[var][method], mode='lines', line=dict(color=cols[j+2], width=7), name=f'No Annealing {method}'))
            error_upper = (noAnnealing_phys_error_mean[var][method] + noAnnealing_phys_error_sem[var][method]).tolist()
            error_lower = (noAnnealing_phys_error_mean[var][method] - noAnnealing_phys_error_sem[var][method]).tolist()
            figs[i+1].add_trace(go.Scatter(x=t[:noAnnealing_max] + t[:noAnnealing_max][::-1], y=error_upper[:noAnnealing_max] + error_lower[:noAnnealing_max][::-1], fill='toself', fillcolor=cols[j+2], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
            
        update_figure(figs[i+1], True)
        # figs[i+1].update_layout(#title=f'{var} error (L1)', 
        #                     xaxis=dict(title='<b>Time (s)</b>', range=[0, dt*(pred_len+1)], nticks=3),
        #                     yaxis=dict(title="<b>L1 error</b>", tick0=0, nticks=6), 
        #                     showlegend=False)
        figs[i+1].update_layout(#title=f'{var} error (L1)', 
                            xaxis=dict(range=[0, dt*(pred_len+1)], nticks=3),
                            yaxis=dict(tick0=0, nticks=6), 
                            showlegend=False)
    
    phys_vars_list = evaluator.get_phys_vars(True)
    for i, f in enumerate(figs):
        f.write_image(os.path.join(save_path, f'{phys_vars_list[i]}.png'), scale=4)
    
    fig = go.Figure()
    for i, method in enumerate(methods):
        fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'Annealing {method}', line=dict(color=cols[i])))
        fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'No Annealing {method}', line=dict(color=cols[2+i])))
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
    fig.write_image(os.path.join(save_path,'legend_only.png'), scale=4)

def summarize_noFilter(args, reject_threshold=0.5):

    save_path = os.path.join(args.output_dir, args.dataset, 'summary', 'noFilter')
    mkdir(save_path)

    pred_len = 60 if args.dataset != 'cylindrical_flow' else 200
    dt = 1/60 if args.dataset != 'cylindrical_flow' else .02
    t = np.linspace(0,dt*pred_len, pred_len).tolist()

    data_path = os.path.join(args.output_dir, args.dataset, 'tasks')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]
    
    smooth_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'regress-smooth-filtered.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        smooth_models.append(name)
    
    noFilter_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'regress-smooth.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        noFilter_models.append(name)
    
    
    
    smooth_losses = {smooth_model: np.load(os.path.join(data_path, smooth_model, 'mlp_predictions/losses.npy' ), allow_pickle=True).item() for smooth_model in smooth_models}
    noFilter_losses = {noFilter_model: np.load(os.path.join(data_path, noFilter_model, 'mlp_predictions/losses.npy' ), allow_pickle=True).item() for noFilter_model in noFilter_models}
    
    evaluator = Physics_Evaluator(args.dataset)
    phys_vars_list = evaluator.get_phys_vars(False)

    figs = [go.Figure() for _ in range(len(phys_vars_list) + 1)]

    all_smooth_losses = {var: [] for var in phys_vars_list}
    all_noFilter_losses = {var: [] for var in phys_vars_list}
    
    smooth_reject = []
    smooth_reject_data = []
    smooth_reject_ratio = {}

    noFilter_reject = []
    noFilter_reject_data = []
    noFilter_reject_ratio = {}


    for smooth_model in smooth_models:
        smooth_reject.append(smooth_losses[smooth_model]['reject'])
        smooth_reject_data.append(smooth_losses[smooth_model]['reject_data'])

        for var in phys_vars_list:
            all_smooth_losses[var].append(smooth_losses[smooth_model][var])
    
    smooth_reject = np.concatenate(smooth_reject)
    smooth_reject_data = np.concatenate(smooth_reject_data)
    for var in phys_vars_list:
            all_smooth_losses[var] = np.concatenate(all_smooth_losses[var])

    smooth_reject_ratio = scale_reject_ratio(pred_len, smooth_reject, smooth_reject_data)
    figs[0].add_trace(go.Scatter(x=t, y=smooth_reject_ratio, line=dict(color=cols[0], width=7), name=f'Filter'))

    for noFilter_model in noFilter_models:
        noFilter_reject.append(noFilter_losses[noFilter_model]['reject'])
        noFilter_reject_data.append(noFilter_losses[noFilter_model]['reject_data'])

        for var in phys_vars_list:
            all_noFilter_losses[var].append(noFilter_losses[noFilter_model][var])
    
    noFilter_reject = np.concatenate(noFilter_reject)
    noFilter_reject_data = np.concatenate(noFilter_reject_data)
    for var in phys_vars_list:
            all_noFilter_losses[var] = np.concatenate(all_noFilter_losses[var])

    noFilter_reject_ratio = scale_reject_ratio(pred_len, noFilter_reject, noFilter_reject_data)
    figs[0].add_trace(go.Scatter(x=t, y=noFilter_reject_ratio, line=dict(color=cols[2], width=7), name=f'No Filter'))

    
        
    update_figure(figs[0], True)
    # figs[0].update_layout( xaxis=dict(title='<b>Time (s)</b>', range=[0, dt*(pred_len+1)], nticks=3),
    #                         yaxis=dict(title='<b>Reject Ratio</b>', range=[-.01,1.05], tick0=0, dtick=0.5), 
    #                         showlegend=False)
    figs[0].update_layout( xaxis=dict(range=[0, dt*(pred_len+1)], nticks=3),
                            yaxis=dict(range=[-.01,1.05], tick0=0, dtick=0.5), 
                            showlegend=False)

    smooth_phys_error_mean = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}
    smooth_phys_error_sem = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}

    noFilter_phys_error_mean = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}
    noFilter_phys_error_sem = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}

    for i, var in enumerate(phys_vars_list):
        smooth_max = pred_len
        
        for p in range(pred_len):
            smooth_error_p = all_smooth_losses[var][:, p]
            smooth_error_p = smooth_error_p[~np.isnan(smooth_error_p)]
            smooth_error_p = remove_outlier(smooth_error_p)
            if smooth_error_p.size / all_smooth_losses[var][:, p].size > reject_threshold:
                smooth_phys_error_mean[var][p] = np.mean(smooth_error_p)
                smooth_phys_error_sem[var][p] = stats.sem(smooth_error_p)
            else:
                smooth_phys_error_mean[var][p] = np.nan
                smooth_phys_error_sem[var][p] = np.nan
                if smooth_max == pred_len:
                    smooth_max = p


        figs[i+1].add_trace(go.Scatter(x=t, y=smooth_phys_error_mean[var], mode='lines', line=dict(color=cols[0], width=7), name=f'Filter'))
        error_upper = (smooth_phys_error_mean[var] + smooth_phys_error_sem[var]).tolist()
        error_lower = (smooth_phys_error_mean[var] - smooth_phys_error_sem[var]).tolist()
        figs[i+1].add_trace(go.Scatter(x=t[:smooth_max] + t[:smooth_max][::-1], y=error_upper[:smooth_max] + error_lower[:smooth_max][::-1], fill='toself', fillcolor=cols[0], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        
        noFilter_max = pred_len
        
        for p in range(pred_len):
            noFilter_error_p = all_noFilter_losses[var][:, p]
            noFilter_error_p = noFilter_error_p[~np.isnan(noFilter_error_p)]
            noFilter_error_p = remove_outlier(noFilter_error_p)
            if noFilter_error_p.size / all_noFilter_losses[var][:, p].size > reject_threshold:
                noFilter_phys_error_mean[var][p] = np.mean(noFilter_error_p)
                noFilter_phys_error_sem[var][p] = stats.sem(noFilter_error_p)
            else:
                noFilter_phys_error_mean[var][p] = np.nan
                noFilter_phys_error_sem[var][p] = np.nan
                if noFilter_max == pred_len:
                    noFilter_max = p


        figs[i+1].add_trace(go.Scatter(x=t, y=noFilter_phys_error_mean[var], mode='lines', line=dict(color=cols[2], width=7), name=f'No Filter'))
        error_upper = (noFilter_phys_error_mean[var] + noFilter_phys_error_sem[var]).tolist()
        error_lower = (noFilter_phys_error_mean[var] - noFilter_phys_error_sem[var]).tolist()
        figs[i+1].add_trace(go.Scatter(x=t[:noFilter_max] + t[:noFilter_max][::-1], y=error_upper[:noFilter_max] + error_lower[:noFilter_max][::-1], fill='toself', fillcolor=cols[2], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        
        update_figure(figs[i+1], True)
        # figs[i+1].update_layout(#title=f'{var} error (L1)', 
        #                     xaxis=dict(title='<b>Time (s)</b>', range=[0, dt*(pred_len+1)], nticks=3),
        #                     yaxis=dict(title="<b>L1 error</b>", tick0=0, nticks=6), 
        #                     showlegend=False)
        figs[i+1].update_layout(#title=f'{var} error (L1)', 
                            xaxis=dict(range=[0, dt*(pred_len+1)], nticks=3),
                            yaxis=dict(tick0=0, nticks=6), 
                            showlegend=False)
    
    phys_vars_list = evaluator.get_phys_vars(True)
    for i, f in enumerate(figs):
        f.write_image(os.path.join(save_path, f'{phys_vars_list[i]}.png'), scale=4)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'Filter', line=dict(color=cols[0])))
    fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'No Filter', line=dict(color=cols[2])))
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
    fig.write_image(os.path.join(save_path,'legend_only.png'), scale=4)

def summarize_discrete(args, reject_threshold=0.5):

    save_path = os.path.join(args.output_dir, args.dataset, 'summary', 'discrete')
    mkdir(save_path)

    pred_len = 60 if args.dataset != 'cylindrical_flow' else 200
    dt = 1/60 if args.dataset != 'cylindrical_flow' else .02
    t = np.linspace(0,dt*pred_len, pred_len).tolist()

    data_path = os.path.join(args.output_dir, args.dataset, 'tasks')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]
    
    smooth_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'regress-smooth-filtered.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        smooth_models.append(name)
    
    discrete_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'discrete-smooth-filtered.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        discrete_models.append(name)
    
    
    
    smooth_losses = {smooth_model: np.load(os.path.join(data_path, smooth_model, 'mlp_predictions/losses.npy' ), allow_pickle=True).item() for smooth_model in smooth_models}
    discrete_losses = {discrete_model: np.load(os.path.join(data_path, discrete_model, 'mlp_predictions/losses.npy' ), allow_pickle=True).item() for discrete_model in discrete_models}
    
    evaluator = Physics_Evaluator(args.dataset)
    phys_vars_list = evaluator.get_phys_vars(False)

    figs = [go.Figure() for _ in range(len(phys_vars_list) + 1)]

    all_smooth_losses = {var: [] for var in phys_vars_list}
    all_discrete_losses = {var: [] for var in phys_vars_list}
    
    smooth_reject = []
    smooth_reject_data = []
    smooth_reject_ratio = {}

    discrete_reject = []
    discrete_reject_data = []
    discrete_reject_ratio = {}


    for smooth_model in smooth_models:
        smooth_reject.append(smooth_losses[smooth_model]['reject'])
        smooth_reject_data.append(smooth_losses[smooth_model]['reject_data'])

        for var in phys_vars_list:
            all_smooth_losses[var].append(smooth_losses[smooth_model][var])
    
    smooth_reject = np.concatenate(smooth_reject)
    smooth_reject_data = np.concatenate(smooth_reject_data)
    for var in phys_vars_list:
            all_smooth_losses[var] = np.concatenate(all_smooth_losses[var])

    smooth_reject_ratio = scale_reject_ratio(pred_len, smooth_reject, smooth_reject_data)
    figs[0].add_trace(go.Scatter(x=t, y=smooth_reject_ratio, line=dict(color=cols[0], width=7), name=f'NeuralODE'))

    for discrete_model in discrete_models:
        discrete_reject.append(discrete_losses[discrete_model]['reject'])
        discrete_reject_data.append(discrete_losses[discrete_model]['reject_data'])

        for var in phys_vars_list:
            all_discrete_losses[var].append(discrete_losses[discrete_model][var])
    
    discrete_reject = np.concatenate(discrete_reject)
    discrete_reject_data = np.concatenate(discrete_reject_data)
    for var in phys_vars_list:
            all_discrete_losses[var] = np.concatenate(all_discrete_losses[var])

    discrete_reject_ratio = scale_reject_ratio(pred_len, discrete_reject, discrete_reject_data)
    figs[0].add_trace(go.Scatter(x=t, y=discrete_reject_ratio, line=dict(color=cols[2], width=7), name=f'Finite Difference'))

    
        
    update_figure(figs[0], True)
    # figs[0].update_layout( xaxis=dict(title='<b>Time (s)</b>', range=[0, dt*(pred_len+1)], nticks=3),
    #                         yaxis=dict(title='<b>Reject Ratio</b>', range=[-.01,1.05], tick0=0, dtick=0.5), 
    #                         showlegend=False)
    figs[0].update_layout( xaxis=dict(range=[0, dt*(pred_len+1)], nticks=3),
                            yaxis=dict(range=[-.01,1.05], tick0=0, dtick=0.5), 
                            showlegend=False)

    smooth_phys_error_mean = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}
    smooth_phys_error_sem = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}

    discrete_phys_error_mean = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}
    discrete_phys_error_sem = {p_var: np.zeros(pred_len) for p_var in phys_vars_list}

    for i, var in enumerate(phys_vars_list):
        smooth_max = pred_len
        
        for p in range(pred_len):
            smooth_error_p = all_smooth_losses[var][:, p]
            smooth_error_p = smooth_error_p[~np.isnan(smooth_error_p)]
            smooth_error_p = remove_outlier(smooth_error_p)
            if smooth_error_p.size / all_smooth_losses[var][:, p].size > reject_threshold:
                smooth_phys_error_mean[var][p] = np.mean(smooth_error_p)
                smooth_phys_error_sem[var][p] = stats.sem(smooth_error_p)
            else:
                smooth_phys_error_mean[var][p] = np.nan
                smooth_phys_error_sem[var][p] = np.nan
                if smooth_max == pred_len:
                    smooth_max = p


        figs[i+1].add_trace(go.Scatter(x=t, y=smooth_phys_error_mean[var], mode='lines', line=dict(color=cols[0], width=7), name=f'NeuralODE'))
        error_upper = (smooth_phys_error_mean[var] + smooth_phys_error_sem[var]).tolist()
        error_lower = (smooth_phys_error_mean[var] - smooth_phys_error_sem[var]).tolist()
        figs[i+1].add_trace(go.Scatter(x=t[:smooth_max] + t[:smooth_max][::-1], y=error_upper[:smooth_max] + error_lower[:smooth_max][::-1], fill='toself', fillcolor=cols[0], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        
        discrete_max = pred_len
        
        for p in range(pred_len):
            discrete_error_p = all_discrete_losses[var][:, p]
            discrete_error_p = discrete_error_p[~np.isnan(discrete_error_p)]
            discrete_error_p = remove_outlier(discrete_error_p)
            if discrete_error_p.size / all_discrete_losses[var][:, p].size > reject_threshold:
                discrete_phys_error_mean[var][p] = np.mean(discrete_error_p)
                discrete_phys_error_sem[var][p] = stats.sem(discrete_error_p)
            else:
                discrete_phys_error_mean[var][p] = np.nan
                discrete_phys_error_sem[var][p] = np.nan
                if discrete_max == pred_len:
                    discrete_max = p


        figs[i+1].add_trace(go.Scatter(x=t, y=discrete_phys_error_mean[var], mode='lines', line=dict(color=cols[2], width=7), name=f'Finite Difference'))
        error_upper = (discrete_phys_error_mean[var] + discrete_phys_error_sem[var]).tolist()
        error_lower = (discrete_phys_error_mean[var] - discrete_phys_error_sem[var]).tolist()
        figs[i+1].add_trace(go.Scatter(x=t[:discrete_max] + t[:discrete_max][::-1], y=error_upper[:discrete_max] + error_lower[:discrete_max][::-1], fill='toself', fillcolor=cols[2], opacity=0.3, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
        
        update_figure(figs[i+1], True)
        # figs[i+1].update_layout(#title=f'{var} error (L1)', 
        #                     xaxis=dict(title='<b>Time (s)</b>', range=[0, dt*(pred_len+1)], nticks=3),
        #                     yaxis=dict(title="<b>L1 error</b>", tick0=0, nticks=6), 
        #                     showlegend=False)
        figs[i+1].update_layout(#title=f'{var} error (L1)', 
                            xaxis=dict(range=[0, dt*(pred_len+1)], nticks=3),
                            yaxis=dict(tick0=0, nticks=6), 
                            showlegend=False)
    
    phys_vars_list = evaluator.get_phys_vars(True)
    for i, f in enumerate(figs):
        f.write_image(os.path.join(save_path, f'{phys_vars_list[i]}.png'), scale=4)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'Neural ODE', line=dict(color=cols[0])))
    fig.add_trace(go.Scatter(x=[0,1,2],y=[0,1,2], visible='legendonly', name=f'Finite Difference', line=dict(color=cols[2])))
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
    fig.write_image(os.path.join(save_path,'legend_only.png'), scale=4)

def summarize_eq(args, model_type='smooth', steps=60):

    mkdir('utils/static')

    data_path = os.path.join(args.output_dir, args.dataset, 'tasks')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]
    
    smooth_models = []
    for seed in seeds:
        if model_type == 'smooth':
            cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'regress-smooth-filtered.yaml'))
        elif model_type == 'noFilter':
            cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'regress-smooth.yaml'))
        else:
            cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'regress-base-filtered.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        smooth_models.append(name)

    show_eq(smooth_models, data_path, args.dataset, steps=steps, port = args.port)

def summarize_smoothness(args,):

    data_path = os.path.join(args.output_dir, args.dataset, 'tasks')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]

    smooth_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'smooth.yaml'))
        
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        smooth_models.append(name)
    
    variation_mean = []
    variation_max = []

    variation_ord2_mean = []
    variation_ord2_max = []

    for smooth_model in smooth_models:

        spline_fitting_path = os.path.join(data_path, smooth_model, 'spline_fitting')

        pre_filter_variation_mean = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_mean.npy'))
        pre_filter_second_variation_mean = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_ord2_mean.npy'))
        pre_filter_variation_max = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_max.npy'))
        pre_filter_second_variation_max = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_ord2_max.npy'))

        variation_mean.extend(pre_filter_variation_mean)
        variation_max.extend(pre_filter_variation_max)

        variation_ord2_mean.extend(pre_filter_second_variation_mean)
        variation_ord2_max.extend(pre_filter_second_variation_max)
    
    variation_mean = np.array(variation_mean)
    variation_max = np.array(variation_max)

    variation_ord2_mean = np.array(variation_ord2_mean)
    variation_ord2_max = np.array(variation_ord2_max)


    print("Smooth")
    print('SM_1_1: %.2f (±%.2f)' % (np.mean(variation_mean), np.std(variation_mean)/np.sqrt(variation_mean.shape[0])))
    print('SM_2_1: %.2f (±%.2f)' % (np.mean(variation_ord2_mean), np.std(variation_ord2_mean)/np.sqrt(variation_ord2_mean.shape[0])))
    print('SM_1_inf mean: %.2f (±%.2f)'  % (np.mean(variation_max), np.std(variation_max)/np.sqrt(variation_max.shape[0])))
    print('SM_2_inf mean: %.2f (±%.2f)'  % (np.mean(variation_ord2_max), np.std(variation_ord2_max)//np.sqrt(variation_ord2_max.shape[0])))

    base_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'base.yaml'))
        
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        base_models.append(name)
    
    variation_mean = []
    variation_max = []

    variation_ord2_mean = []
    variation_ord2_max = []

    for base_model in base_models:

        spline_fitting_path = os.path.join(data_path, base_model, 'spline_fitting')

        pre_filter_variation_mean = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_mean.npy'))
        pre_filter_second_variation_mean = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_ord2_mean.npy'))
        pre_filter_variation_max = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_max.npy'))
        pre_filter_second_variation_max = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_ord2_max.npy'))

        variation_mean.extend(pre_filter_variation_mean)
        variation_max.extend(pre_filter_variation_max)

        variation_ord2_mean.extend(pre_filter_second_variation_mean)
        variation_ord2_max.extend(pre_filter_second_variation_max)
    
    variation_mean = np.array(variation_mean)
    variation_max = np.array(variation_max)

    variation_ord2_mean = np.array(variation_ord2_mean)
    variation_ord2_max = np.array(variation_ord2_max)


    print("Base")
    print('SM_1_1: %.2f (±%.2f)' % (np.mean(variation_mean), np.std(variation_mean)/np.sqrt(variation_mean.shape[0])))
    print('SM_2_1: %.2f (±%.2f)' % (np.mean(variation_ord2_mean), np.std(variation_ord2_mean)/np.sqrt(variation_ord2_mean.shape[0])))
    print('SM_1_inf mean: %.2f (±%.2f)' % (np.mean(variation_max), np.std(variation_max)/np.sqrt(variation_max.shape[0])))
    print('SM_2_inf mean: %.2f (±%.2f)' % (np.mean(variation_ord2_max), np.std(variation_ord2_max)//np.sqrt(variation_ord2_max.shape[0])))


def summarize_id(args):
    data_path = os.path.join(args.output_dir, args.dataset, 'variables')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]

    id = []

    for seed in seeds:
        id.append(np.load(os.path.join(data_path, f'encoder-decoder_{seed}/intrinsic_dimension.npy')))
        id.append(np.load(os.path.join(data_path, f'encoder-decoder-64_{seed}/intrinsic_dimension.npy')))
    
    id = np.concatenate(id)

    print(f'System: {args.dataset}\nIntrinsic Dimension Estimate: {id.mean():.2f} (± {id.std()/np.sqrt(id.shape[0]):.2f})')

def summarize_nsv_singleStep(args):
    data_path = os.path.join(args.output_dir, args.dataset, 'tasks')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]

    smooth_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'smooth.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        smooth_models.append(name)
    
    noAnnealing_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'smooth-noAnnealing.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        noAnnealing_models.append(name)


    base_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'base.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        base_models.append(name)

    base_accuracy = []

    for base_model in base_models:

        result = np.load(os.path.join(data_path, base_model, 'test_result/results.npy'), allow_pickle=True).item()

        base_accuracy.append(result['pxl_rec_test_loss_epoch'])
    
    base_accuracy = np.array(base_accuracy)

    print(f'{args.dataset} base average pxl rec loss: ', '%.3e (±%.3e)' % (base_accuracy.mean(), base_accuracy.std()/np.sqrt(base_accuracy.shape[0])))

    smooth_accuracy = []

    for smooth_model in smooth_models:

        result = np.load(os.path.join(data_path, smooth_model, 'test_result/results.npy'), allow_pickle=True).item()

        smooth_accuracy.append(result['pxl_rec_test_loss_epoch'])
    
    smooth_accuracy = np.array(smooth_accuracy)

    print(f'{args.dataset} smooth average pxl rec loss: ', '%.3e (±%.3e)' % (smooth_accuracy.mean(), smooth_accuracy.std()/np.sqrt(smooth_accuracy.shape[0])))
    
    noAnnealing_accuracy = []

    for noAnnealing_model in noAnnealing_models:

        result = np.load(os.path.join(data_path, noAnnealing_model, 'test_result/results.npy'), allow_pickle=True).item()

        noAnnealing_accuracy.append(result['pxl_rec_test_loss_epoch'])
    
    noAnnealing_accuracy = np.array(noAnnealing_accuracy)

    print(f'{args.dataset} no annealing average pxl rec loss: ', '%.3e (±%.3e)' % (noAnnealing_accuracy.mean(), noAnnealing_accuracy.std()/np.sqrt(noAnnealing_accuracy.shape[0])))
    


def summarize_nsvf_singleStep(args):
    data_path = os.path.join(args.output_dir, args.dataset, 'tasks')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]

    base_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'regress-base-filtered.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        base_models.append(name)


    smooth_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'regress-smooth-filtered.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        smooth_models.append(name)
    
    noFilter_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'regress-smooth.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        noFilter_models.append(name)
    
    discrete_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'discrete-smooth-filtered.yaml'))
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        discrete_models.append(name)

    
    base_accuracy = []

    for base_model in base_models:

        result = np.load(os.path.join(data_path, base_model, 'test_result/results.npy'), allow_pickle=True).item()

        base_accuracy.append(result['pxl_rec_test_loss_epoch'])
    base_accuracy = np.array(base_accuracy)

    print(f'{args.dataset} base average pxl rec loss: ', '%.3e (±%.3e)' % (base_accuracy.mean(), base_accuracy.std()/np.sqrt(base_accuracy.shape[0])))

    smooth_accuracy = []

    for smooth_model in smooth_models:

        result = np.load(os.path.join(data_path, smooth_model, 'test_result/results.npy'), allow_pickle=True).item()

        smooth_accuracy.append(result['pxl_rec_test_loss_epoch'])
    smooth_accuracy = np.array(smooth_accuracy)

    print(f'{args.dataset} smooth average pxl rec loss: ', '%.3e (±%.3e)' % (smooth_accuracy.mean(), smooth_accuracy.std()/np.sqrt(smooth_accuracy.shape[0])))


    noFilter_accuracy = []

    for noFilter_model in noFilter_models:

        result = np.load(os.path.join(data_path, noFilter_model, 'test_result/results.npy'), allow_pickle=True).item()

        noFilter_accuracy.append(result['pxl_rec_test_loss_epoch'])
    
    noFilter_accuracy = np.array(noFilter_accuracy)

    print(f'{args.dataset} noFilter smooth average pxl rec loss: ', '%.3e (±%.3e)' % (noFilter_accuracy.mean(), noFilter_accuracy.std()/np.sqrt(noFilter_accuracy.shape[0])))
    
    discrete_accuracy = []

    for discrete_model in discrete_models:

        result = np.load(os.path.join(data_path, discrete_model, 'test_result/results.npy'), allow_pickle=True).item()

        discrete_accuracy.append(result['pxl_rec_test_loss_epoch'])
    
    discrete_accuracy = np.array(discrete_accuracy)

    print(f'{args.dataset} (finite difference training) smooth average pxl rec loss: ', '%.3e (±%.3e)' % (discrete_accuracy.mean(), discrete_accuracy.std()/np.sqrt(discrete_accuracy.shape[0])))
    

def filter_remaining(args):

    data_path = os.path.join(args.output_dir, args.dataset, 'tasks')

    seeds = [1,2,3] if args.dataset != 'spring_mass' else [1,3,4]

    smooth_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'smooth.yaml'))

        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        smooth_models.append(name)



    sufs = ['_train', '_val', '']
    print("Smooth")
    for suf in sufs:

        totals = []
        remaining = []

        for smooth_model in smooth_models:

            total = np.load(os.path.join(args.output_dir, args.dataset, 'variables'+suf, smooth_model, 'total.npy'))
            invalid = np.load(os.path.join(args.output_dir, args.dataset, 'variables'+suf, smooth_model, 'invalid.npy'))
            filenames = os.listdir(os.path.join(args.output_dir, args.dataset, 'tasks', smooth_model, 'nsv_trajectories'))

            totals.append(len(total))
            remaining.append(len(total) - len(invalid))

        totals = np.array(totals)
        remaining = np.array(remaining)
        print('Remaining Trajectories (%s) : %.2f (±%.2f) /  %d' % (suf[1:] if suf!='' else 'test', np.mean(remaining), np.std(remaining)/np.sqrt(remaining.shape[0]),np.mean(totals)))
    
    base_models = []
    for seed in seeds:
        cfg = load_config(filepath=os.path.join(args.config_dir, args.dataset, f'trial{seed}', 'base.yaml'))
        
        cfg_args = munchify(cfg)
        name = create_name(cfg_args)
        base_models.append(name)



    sufs = ['_train', '_val', '']
    print("Base")

    for suf in sufs:

        totals = []
        remaining = []

        for base_model in base_models:

            total = np.load(os.path.join(args.output_dir, args.dataset, 'variables'+suf, base_model, 'total.npy'))
            invalid = np.load(os.path.join(args.output_dir, args.dataset, 'variables'+suf, base_model, 'invalid.npy'))
            filenames = os.listdir(os.path.join(args.output_dir, args.dataset, 'tasks', base_model, 'nsv_trajectories'))

            totals.append(len(total))
            remaining.append(len(total) - len(invalid))

        totals = np.array(totals)
        remaining = np.array(remaining)
        print('Remaining Trajectories (%s) : %.2f (±%.2f) /  %d' % (suf[1:] if suf!='' else 'test', np.mean(remaining), np.std(remaining)/np.sqrt(remaining.shape[0]),np.mean(totals)))
        


def main():

    parser = argparse.ArgumentParser(description='Summarize All Seeds')

    # Mode for script
    parser.add_argument('-mode', help='summarize mode: nsv or nsvf',
                    type=str, required=False, default='nsv')
    parser.add_argument('-model_type', help='smooth, base or noFilter',
                    type=str, required=False, default='smooth')
    parser.add_argument('-output_dir', help='output directory',
                    type=str, required=False, default='outputs')
    parser.add_argument('-config_dir', help='config directory',
                    type=str, required=False, default='configs')
    parser.add_argument('-dataset', help='dataset',
                        type=str, required=True)
    parser.add_argument('-steps', help='dataset',
                        type=int, required=False, default=60)
    parser.add_argument('-port', help='port number',
                    type=int, required=False, default=8002)

    args = parser.parse_args()

    if args.mode == 'nsv':
        summarize_nsv(args)
    elif args.mode == 'nsv_single':
        summarize_nsv_singleStep(args)
    if args.mode == 'noAnnealing':
        summarize_noAnnealing(args)
    elif args.mode == 'nsvf':
        summarize_nsvf(args)
    elif args.mode == 'nsvf_single':
        summarize_nsvf_singleStep(args)
    if args.mode == 'noFilter':
        summarize_noFilter(args)
    if args.mode == 'discrete':
        summarize_discrete(args)
    elif args.mode == 'eq':
        summarize_eq(args, args.model_type)
    elif args.mode == 'smoothness':
        summarize_smoothness(args)
    elif args.mode == 'id':
        summarize_id(args)
    if args.mode == 'filter':
        filter_remaining(args)


if __name__ == '__main__':
    
    main()