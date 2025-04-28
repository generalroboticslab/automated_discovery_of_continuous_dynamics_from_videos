from flask import Flask, render_template
import os
from tqdm import tqdm
from utils.misc import *
import cv2 
from .analysis import Physics_Evaluator

def generate_video_directory(fig_path, path_nums, flag="", fps=60, delete_after=False):

    vid_folder = os.path.join(fig_path, flag+'videos') if not delete_after else fig_path
    if not delete_after:
        mkdir(vid_folder)

    print("Creating Videos")
    for i in tqdm(path_nums):

        generate_video(fig_path, str(i), os.path.join(vid_folder, f'{i}.mp4'), fps=fps, delete_after=delete_after)


def generate_video(fig_path, path_num, dst_path, fps=60, delete_after=False):

    fig_path = os.path.join(fig_path, path_num)

    images = [img for img in os.listdir(fig_path)
                if img.endswith(".png") or img.endswith(".jpg")]
    
    images.sort(key = lambda x: int(x[:-4]))

    frame = cv2.imread(os.path.join(fig_path, images[0]))

    height, width, layers = frame.shape  

    video = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) 

    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(fig_path, image))) 
    
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated

    if delete_after:
        shutil.rmtree(fig_path)

cols = ['#DF1E1E', '#F0975A', '#1B66CB', '#149650', '#6F4E37', '#912BBC']
transparent_cols = ['rgb(223, 30, 30, 0.1)', 'rgb(240, 151, 90, 0.1)', 'rgb(27, 102, 203, 0.2)', 'rgb(20, 150, 80, 0.2)', 'rgb(111, 78, 55, 0.2)', 'rgb(145, 43, 188, 0.2)']

colorscale=[[0.0, "rgb(49,54,149)"],
            [0.1111111111111111, "rgb(69,117,180)"],
            [0.2222222222222222, "rgb(116,173,209)"],
            [0.3333333333333333, "rgb(171,217,233)"],
            [0.4444444444444444, "rgb(224,243,248)"],
            [0.5555555555555556, "rgb(254,224,144)"],
            [0.6666666666666666, "rgb(253,174,97)"],
            [0.7777777777777778, "rgb(244,109,67)"],
            [0.8888888888888888, "rgb(215,48,39)"],
            [1.0, "rgb(165,0,38)"]]

def update_figure(fig, small_margin=False):
    fig.update_xaxes(showgrid=False, tickfont=dict(family="Helvetica Neue", size=36, color="black"))
    fig.update_yaxes(showgrid=False, tickfont=dict(family="Helvetica Neue", size=36, color="black"))
    fig.update_xaxes(showline=True, linewidth=5, linecolor="black", ticks='outside', tickwidth=5, tickprefix="<b>",ticksuffix ="</b>")
    fig.update_yaxes(showline=True, linewidth=5, linecolor="black", ticks='outside', tickwidth=5, tickprefix="<b>",ticksuffix ="</b>")
    fig.update_layout(legend=go.layout.Legend(traceorder="normal",
                                              font=dict(family="Helvetica Neue", size=36, color="black")))
    fig.update_layout(font=dict(family="Helvetica Neue", size=36, color="black"),
                      paper_bgcolor='rgba(255,255,255,255)',
                      plot_bgcolor='rgba(255,255,255,255)'
                     )
    if small_margin:
        fig.update_layout(margin=dict(l=50, r=30, t=20, b=20))

def update_figure_3d(fig, small_margin=False):


    fig.update_layout(legend=go.layout.Legend(traceorder="normal",
                                              font=dict(family="Helvetica Neue", size=18, color="black"),
                    ))
    fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgba(255,255,255,255)",
                         linecolor="black",
                         gridcolor="black",
                         showbackground=True,
                         linewidth=5,
                         tickfont = dict(family="Helvetica Neue", size=18),
                         ticks="outside", ticklen=35, tickwidth=5,tickprefix="<b>",ticksuffix ="</b>"),
                    yaxis = dict(
                        backgroundcolor="rgba(255,255,255,255)",
                        linecolor="black",
                        gridcolor="black",
                        showbackground=True,
                        linewidth=5,
                        tickfont = dict(family="Helvetica Neue", size=18),
                        ticks="outside", ticklen=35, tickwidth=5,tickprefix="<b>",ticksuffix ="</b>"),
                    zaxis = dict(
                        backgroundcolor="rgba(255,255,255,255)",
                        linecolor="black",
                        gridcolor="black",
                        showbackground=True,
                        linewidth=5,
                        tickfont = dict(family="Helvetica Neue", size=18),
                        ticks="outside", ticklen=50, tickwidth=5,tickprefix="<b>",ticksuffix ="</b>")),
                    font=dict(family="Helvetica Neue", size=28, color="black"),
                    paper_bgcolor='rgba(255,255,255,255)',
                    plot_bgcolor='rgba(255,255,255,255)',
                    scene_camera = dict(eye=dict(x=1.5, y=1.5, z=1.2), center=dict(x=0, y=0, z=-0.3)),
                    margin=dict(l=50, r=30, t=20, b=20)
                     )
    if small_margin:
        fig.update_layout(margin=dict(l=50, r=30, t=20, b=20))
        
def update_figure_small(fig, small_margin=False):
    fig.update_xaxes(showgrid=False, tickfont=dict(family="Helvetica Neue", size=18, color="black"),tickprefix="<b>",ticksuffix ="</b>")
    fig.update_yaxes(showgrid=False, tickfont=dict(family="Helvetica Neue", size=18, color="black"),tickprefix="<b>",ticksuffix ="</b>")
    fig.update_xaxes(showline=True, linewidth=3, linecolor="black", ticks='outside', tickwidth=3, tickprefix="<b>",ticksuffix ="</b>")
    fig.update_yaxes(showline=True, linewidth=3, linecolor="black", ticks='outside', tickwidth=3, tickprefix="<b>",ticksuffix ="</b>")
    fig.update_layout(legend=go.layout.Legend(traceorder="normal",
                                              font=dict(family="Helvetica Neue", size=18, color="black"),
                    ))
    fig.update_layout(font=dict(family="Helvetica Neue", size=18, color="black"),
                      paper_bgcolor='rgba(255,255,255,255)',
                      plot_bgcolor='rgba(255,255,255,255)',
                     )
    if small_margin:
        fig.update_layout(margin=dict(l=50, r=30, t=20, b=20))

def show_eq(smooth_models, data_path, dataset, steps=60, port=8002):
    
    eqs = {}

    for i, smooth_model in enumerate(smooth_models):

        eq_path = os.path.join(data_path, smooth_model, f'mlp_equilibrium')
        eqs[i] = np.load(os.path.join(eq_path, 'eq_points.npy'), allow_pickle=True).item()

        num_eq = len(eqs[i]['validity'])
        n = 0
        roots = []
        guesses = []
        jacobians = []
        eigenValues = []
        distances = []
        stabilities = []
        delta_per_epsilon = []
        validity = []
        for j in range(num_eq):

            if eqs[i]['successes'][j] == True and eqs[i]['validity'][j] == True:
                # frame 1
                src_path = os.path.join(eq_path, f'{j}/m_0.png')
                dst_path = os.path.join('utils/static', f'eq_{i}_{n}_0.png')
                os.system(f'cp {src_path} {dst_path}')

                # frame 2
                src_path = os.path.join(eq_path, f'{j}/m_1.png')
                dst_path = os.path.join('utils/static', f'eq_{i}_{n}_1.png')
                os.system(f'cp {src_path} {dst_path}')

                roots.append(eqs[i]['roots'][j])
                guesses.append(eqs[i]['guesses'][j])
                jacobians.append(eqs[i]['jacobians'][j])
                eigenValues.append(eqs[i]['eigenValues'][j])
                stabilities.append(eqs[i]['stabilities'][j])
                delta_per_epsilon.append(eqs[i]['delta_per_epsilon'][j])
                distances.append(np.array(eqs[i]['distances'][j]))

                n += 1
            
        eqs[i]['num_eq'] = n
        if n != 0:
            eqs[i]['roots'] = np.array(roots)
            eqs[i]['guesses'] = np.array(guesses)
            eqs[i]['jacobians'] = np.array(jacobians)
            eqs[i]['eigenValues'] = np.array(eigenValues)
            eqs[i]['stabilities'] = np.array(stabilities)
            eqs[i]['delta_per_epsilon'] = delta_per_epsilon
            eqs[i]['distances'] = np.stack(distances, axis=0)
            eqs[i]['mlp_eq_distances_mean'] = np.mean(eqs[i]['distances'], axis=2)
            eqs[i]['mlp_eq_distances_max'] = np.max(eqs[i]['distances'], axis=2)
    
    app = Flask(__name__)

    @app.route('/')
    def index():
    
        return render_template('results_eq.html', dataset=dataset, eqs=eqs, smooth_models=smooth_models)
    
    app.run(debug=True, port = port)



def show_nsvf(args):

    output_dir = args.output_dir
    dataset = args.dataset
    model_name = create_name(args)
    seed = args.seed
    num_dims = get_experiment_dim(dataset, 0)

    regress_name = create_name(args)
    regress_path = os.path.join(output_dir, dataset, 'tasks', f'{regress_name}')

    invalid = np.load(os.path.join(output_dir, dataset, 'tasks', args.nsv_model_name, 'invalid.npy'))

    rng = np.random.default_rng(seed)
    trajectories_path = os.path.join(output_dir, dataset, 'tasks', args.nsv_model_name, 'nsv_trajectories')
    filenames = rng.permutation(os.listdir(trajectories_path))

    if 'filtered' in model_name:
        filtered_filenames = []
        for i in range(len(filenames)):
            if int(filenames[i][:-4]) not in invalid:
                filtered_filenames.append(filenames[i])
    else:
        filtered_filenames= filenames

    regress_test_result = np.load(os.path.join(regress_path, 'test_result', 'results.npy'), allow_pickle=True).item()

    # NSVF Sample Trajectories
    mlp_trajectories_path = os.path.join(regress_path, 'mlp_predictions', 'trajectories')
    mlp_time_series_path = os.path.join(regress_path, 'mlp_predictions', 'time_series')
    for i in range(15):

        # mlp trajectory
        src_path = os.path.join(mlp_trajectories_path, filtered_filenames[i])
        dst_path = os.path.join('utils/static', f'mlp_traj_{i}.png')
        os.system(f'cp {src_path} {dst_path}')

        # mlp time series
        src_path = os.path.join(mlp_time_series_path, filtered_filenames[i])
        dst_path = os.path.join('utils/static', f'mlp_time_series_{i}.png')
        os.system(f'cp {src_path} {dst_path}')

    mlp_visualizations = []
    # NSVF Visualizations
    mlp_visualizations_path = os.path.join(regress_path, 'mlp_visualization')
    for i in range(num_dims):

        # mlp pred
        src_path = os.path.join(mlp_visualizations_path, f'pred_{i+1}.png')
        dst_path = os.path.join('utils/static', f'mlp_visualization_{2*i}.png')
        os.system(f'cp {src_path} {dst_path}')
        mlp_visualizations.append(f'dV{i+1}/dt')

        # mlp target
        src_path = os.path.join(mlp_visualizations_path, f'tar_{i+1}.png')
        dst_path = os.path.join('utils/static', f'mlp_visualization_{2*i+1}.png')
        os.system(f'cp {src_path} {dst_path}')
        mlp_visualizations.append(f'(V{i+1}(t+dt)-V{i+1}(t))/dt')
    
    if num_dims == 2:
        src_path = os.path.join(mlp_visualizations_path, f'gradient_field.png')
        dst_path = os.path.join('utils/static', f'mlp_visualization_{2*num_dims}.png')
        os.system(f'cp {src_path} {dst_path}')
        mlp_visualizations.append('Gradient Field')
    if num_dims == 4:
        src_path = os.path.join(mlp_visualizations_path, f'gradient_field_v1v2.png')
        dst_path = os.path.join('utils/static', f'mlp_visualization_{2*num_dims}.png')
        os.system(f'cp {src_path} {dst_path}')
        mlp_visualizations.append('Gradient Field V1 V2')

        src_path = os.path.join(mlp_visualizations_path, f'gradient_field_v3v4.png')
        dst_path = os.path.join('utils/static', f'mlp_visualization_{2*num_dims+1}.png')
        os.system(f'cp {src_path} {dst_path}')
        mlp_visualizations.append('Gradient Field V3 V4')
    if num_dims == 3:
        src_path = os.path.join(mlp_visualizations_path, f'gradient_field_v1v2.png')
        dst_path = os.path.join('utils/static', f'mlp_visualization_{2*num_dims}.png')
        os.system(f'cp {src_path} {dst_path}')
        mlp_visualizations.append('Gradient Field V1 V2')

        src_path = os.path.join(mlp_visualizations_path, f'gradient_field_v1v3.png')
        dst_path = os.path.join('utils/static', f'mlp_visualization_{2*num_dims+1}.png')
        os.system(f'cp {src_path} {dst_path}')
        mlp_visualizations.append('Gradient Field V1 V3')

        src_path = os.path.join(mlp_visualizations_path, f'gradient_field_v2v3.png')
        dst_path = os.path.join('utils/static', f'mlp_visualization_{2*num_dims+2}.png')
        os.system(f'cp {src_path} {dst_path}')
        mlp_visualizations.append('Gradient Field V2 V3')

    
    # NSVF long term accuracy
    mlp_long_term_prediction_path = os.path.join(regress_path, 'mlp_predictions', 'plots')
    if os.path.exists(mlp_long_term_prediction_path):

        src_path = os.path.join(mlp_long_term_prediction_path, 'all.png')
        print(src_path)
        dst_path = os.path.join('utils/static', f'mlp_long_term_prediction.png')
        #os.system(f'touch {dst_path}')
        os.system(f'cp \"{src_path}\" \"{dst_path}\"')
    

    # NSVF Equilibrium
    eq_path = os.path.join(regress_path, 'mlp_equilibrium')
    eq_points = np.load(os.path.join(eq_path, 'eq_points.npy'), allow_pickle=True).item()

    num_eq = len(eq_points['validity'])
    for i in range(num_eq):

        # frame 1
        src_path = os.path.join(eq_path, f'{i}/m_0.png')
        dst_path = os.path.join('utils/static', f'eq_{i}_0.png')
        os.system(f'cp {src_path} {dst_path}')

        # frame 2
        src_path = os.path.join(eq_path, f'{i}/m_1.png')
        dst_path = os.path.join('utils/static', f'eq_{i}_1.png')
        os.system(f'cp {src_path} {dst_path}')

    app = Flask(__name__)

    @app.route('/')
    def index():
            
        results = {}
        ignored = ['data_filepath', 'log_dir', 'num_workers', 'model_name', 'if_test', 'if_cuda', 'reconstruct_loss_weight', 'latent_model_name', 'num_gpus',
                'train_batch', 'test_batch', 'val_batch', 'architecture', 'output_dir', 'method', 'positive_range', 'negative_range', 'inference_mode']
        results['hparams'] = dict((k, args[k]) for k in args.keys() if k not in ignored)

        results['total'] = len(filenames)
        results['num_filtered'] = len(filtered_filenames)
        results['filenames'] = filtered_filenames[:15]
        results['mlp_visualizations'] = mlp_visualizations

        if num_dims == 2:
            results['num_mlp_visualizations'] = 2*num_dims + 1
        elif num_dims == 4:
            results['num_mlp_visualizations'] = 2*num_dims + 2
        else:
            results['num_mlp_visualizations'] = 2*num_dims + 3

        results['num_eq'] = num_eq
        if os.path.exists(regress_path):
            results['mlp_rec_test_loss'] = regress_test_result['rec_test_loss_epoch']
            results['mlp_pxl_rec_test_loss_epoch'] = regress_test_result['pxl_rec_test_loss_epoch']

            results['eq_points'] = eq_points
            results['mlp_eq_distances_mean'] = np.mean(eq_points['distances'], axis=2)
            results['mlp_eq_distances_max'] = np.max(eq_points['distances'], axis=2)
    
        return render_template('results_nsvf.html', model_name=model_name, results=results)
    
    app.run(debug=True, port = args.port)

def show_nsv(args):

    output_dir = args.output_dir
    dataset = args.dataset
    model_name = create_name(args)
    seed = args.seed
    num_dims = get_experiment_dim(dataset, 0)

    regress_name = "regress" if dataset in ['single_pendulum', 'spring_mass'] else "regressDeeper"

    mkdir('utils/static')


    # spline fitting results

    spline_fitting_path = os.path.join(output_dir, dataset, 'tasks', model_name, 'spline_fitting')

    pre_filter_variation_mean = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_mean.npy'))
    pre_filter_second_variation_mean = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_ord2_mean.npy'))
    pre_filter_deviation_arr = np.load(os.path.join(spline_fitting_path, 'pre_filter_deviation.npy'))
    pre_filter_tangling_arr = np.load(os.path.join(spline_fitting_path, 'pre_filter_tangling.npy'))
    pre_filter_tangling_max_arr = np.load(os.path.join(spline_fitting_path, 'pre_filter_tangling_max.npy'))
    pre_filter_variation_max = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_max.npy'))
    pre_filter_second_variation_max = np.load(os.path.join(spline_fitting_path, 'pre_filter_variation_ord2_max.npy'))

    post_filter_deviation_arr = np.load(os.path.join(spline_fitting_path, 'post_filter_deviation.npy'))
    post_filter_tangling_arr = np.load(os.path.join(spline_fitting_path, 'post_filter_tangling.npy'))
    post_filter_tangling_max_arr = np.load(os.path.join(spline_fitting_path, 'post_filter_tangling_max.npy'))

    test_result = np.load(os.path.join(output_dir, dataset, 'tasks', model_name, 'test_result', 'results.npy'), allow_pickle=True).item()
    print(test_result)
    regress_test_result_filepath = os.path.join(output_dir, dataset, 'tasks', f'{regress_name}_{seed}_'+ model_name + '_filtered', 'test_result', 'results.npy')
    if os.path.exists(regress_test_result_filepath):
        regress_test_result = np.load(regress_test_result_filepath, allow_pickle=True).item()
    else:
        regress_test_result = None

    rng = np.random.default_rng(seed)

    # NSV Sample Trajectories
    trajectories_path = os.path.join(output_dir, dataset, 'tasks', model_name, 'nsv_trajectories')
    time_series_path = os.path.join(output_dir, dataset, 'tasks', model_name, 'time_series')
    first_order_path = os.path.join(output_dir, dataset, 'tasks', model_name, 'first_order_derivatives')
    invalid = np.load(os.path.join(output_dir, dataset, 'tasks', model_name, 'invalid.npy'))
    
    filenames = rng.permutation(os.listdir(trajectories_path))
    filtered_filenames = []
    for i in range(len(filenames)):
        if int(filenames[i][:-4]) not in invalid:
            filtered_filenames.append(filenames[i])

    for i in range(15):
        # nsv trajectory
        src_path = os.path.join(trajectories_path, filenames[i])
        dst_path = os.path.join('utils/static', f'traj_{i}.png')
        os.system(f'cp {src_path} {dst_path}')

        # time series
        src_path = os.path.join(time_series_path, filenames[i])
        dst_path = os.path.join('utils/static', f'time_series_{i}.png')
        os.system(f'cp {src_path} {dst_path}')

        # first_order_derivative
        src_path = os.path.join(first_order_path, filenames[i])
        dst_path = os.path.join('utils/static', f'first_order_{i}.png')
        os.system(f'cp {src_path} {dst_path}')

        # filtered nsv trajectory
        src_path = os.path.join(trajectories_path, filtered_filenames[i])
        dst_path = os.path.join('utils/static', f'filtered_traj_{i}.png')
        os.system(f'cp {src_path} {dst_path}')

        # filtered time series
        src_path = os.path.join(time_series_path, filtered_filenames[i])
        dst_path = os.path.join('utils/static', f'filtered_time_series_{i}.png')
        os.system(f'cp {src_path} {dst_path}')

        # filtered first_order_derivative
        src_path = os.path.join(first_order_path, filtered_filenames[i])
        dst_path = os.path.join('utils/static', f'filtered_first_order_{i}.png')
        os.system(f'cp {src_path} {dst_path}')
    
    # NSV Visualizations
    embedding_path = os.path.join(output_dir, dataset, 'tasks', model_name, 'nsv_embedding')
    phys_estimator = Physics_Evaluator(dataset)
    var_list = [var + ' (t=2)' for var in phys_estimator.get_phys_vars()]

    for i, var in enumerate(var_list):
        src_path = os.path.join(embedding_path, var+'.png')
        print(src_path)
        dst_path = os.path.join('utils/static', f'embed_{i}.png')
        os.system(f'cp "{src_path}" {dst_path}')

    # NSV long term accuracy
    long_term_prediction_path = os.path.join(output_dir, dataset, 'predictions_long_term', model_name, 'plots')
    if os.path.exists(long_term_prediction_path):

        src_path = os.path.join(long_term_prediction_path, 'all.png')
        print(src_path)
        dst_path = os.path.join('utils/static', f'long_term_prediction.png')
        os.system(f'cp \"{src_path}\" \"{dst_path}\"')

    app = Flask(__name__)

    @app.route('/')
    def index():
            
        results = {}
        ignored = ['data_filepath', 'log_dir', 'num_workers', 'model_name',  'if_test', 'if_cuda',  'num_gpus',
                'train_batch', 'test_batch', 'val_batch', 'architecture', 'output_dir', 'method', 'positive_range', 'negative_range', 'inference_mode']
        results['hparams'] = dict((k, args[k]) for k in args.keys() if k not in ignored)

        results['pre_filter_variation'] = 'pre filter variation mean: %.2f (±%.2f)' % (np.mean(pre_filter_variation_mean), np.std(pre_filter_variation_mean))
        results['pre_filter_variation_ord2'] = 'pre filter 2nd order variation mean: %.2f (±%.2f)' % (np.mean(pre_filter_second_variation_mean), np.std(pre_filter_second_variation_mean))
        results['pre_filter_deviation'] = 'pre filter trajectory deviation mean: %.2f' % (np.mean(pre_filter_deviation_arr))
        results['pre_filter_tangling'] = 'pre filter tangling mean: %.2f' % (np.mean(pre_filter_tangling_arr))
        results['pre_filter_tangling_max'] = 'pre filter tangling max: %.2f' % (np.max(pre_filter_tangling_max_arr))
        results['pre_filter_variation_max_max'] = 'pre filter max max variation: %.2f' % np.max(pre_filter_variation_max)
        results['pre_filter_variation_ord2_max_max'] = 'pre filter max max 2nd order variation: %.2f' % np.max(pre_filter_second_variation_max)
        results['pre_filter_variation_mean_max'] = 'pre filter mean max variation: %.2f' % np.mean(pre_filter_variation_max)
        results['pre_filter_variation_ord2_mean_max'] = 'pre filter mean max 2nd order variation: %.2f' % np.mean(pre_filter_second_variation_max)
        
        results['post_filter_deviation'] = 'post filter trajectory deviation mean: %.2f' % (np.mean(post_filter_deviation_arr))
        results['post_filter_tangling'] = 'post filter tangling mean: %.2f' % (np.mean(post_filter_tangling_arr))
        results['post_filter_tangling_max'] = 'post filter tangling max: %.2f' % (np.max(post_filter_tangling_max_arr))
        results['filenames'] = filenames[:15]
        results['filtered_filenames'] = filtered_filenames[:15]

        results['num_vars'] = len(var_list)
        results['var_list'] = var_list
        results['total'] = len(filenames)
        results['num_filtered'] = len(filtered_filenames)

        results['pxl_rec_test_loss_epoch'] = test_result['pxl_rec_test_loss_epoch']
    
        return render_template('results_nsv.html', model_name=model_name, results=results)
    
    app.run(debug=True, port = args.port)
