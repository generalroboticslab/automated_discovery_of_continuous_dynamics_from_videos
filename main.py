import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pprint
import argparse
from munch import munchify

from models.vis_dynamics_model import *
from models.data_module import *
from models.callbacks import *

from utils.pred import *
from utils.misc import *
from utils.show import *

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint



def prepare_Trainer(args):

    output_path = os.path.join(os.getcwd(), args.output_dir, args.dataset)
    model_name = create_name(args)

    custom_progress_bar = LitProgressBar()

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path +  "/checkpoints/" + model_name,
        verbose=True,
        monitor='val_loss',
        mode='min',
        auto_insert_metric_name=True,
        save_last=True)
    
    cyclic_annealing_callback = CyclicalAnnealingCallback(**args)
    
    callbacks = [custom_progress_bar, checkpoint_callback, cyclic_annealing_callback]
    
    if 'encoder-decoder' in args.model_name:
        post_process = IntrinsicDimensionEstimator()
        callbacks.append(post_process)
    elif 'base' in args.model_name:
        post_process = SmoothnessEvaluator()
        callbacks.append(post_process)
    elif 'smooth' in args.model_name:
        post_process = SmoothnessEvaluator()
        callbacks.append(post_process)

    logger = pl_loggers.WandbLogger(save_dir=output_path, 
                                        name=model_name, 
                                        log_model=False,
                                        project=args.dataset,
                                        resume="allow")

    trainer = Trainer(devices=args.num_gpus,
                      max_epochs=args.epochs,
                      deterministic=True,
                      accelerator='gpu',
                      default_root_dir=output_path,
                      val_check_interval=1.0,
                      callbacks=callbacks,
                      logger=logger,
                      **get_validArgs(Trainer, args))

    return trainer

def prepare_DataModule(args):

    dm = SimulationDataModule(**args)

    return dm

def prepare_Model(args, is_test):

    if 'smooth' in args.model_name:
        model = SmoothNSVAutoencoder(**args)
    elif 'base' in args.model_name:
        model = NSVAutoencoder(**args)
    elif 'encoder-decoder-64' in args.model_name:
        model = LatentAutoEncoder(3, **args)
    elif 'encoder-decoder' in args.model_name:
        model = LargeLatentAutoEncoder(3, **args)
    else:
        exit("Invalid Model Name")
    
    if is_test:

        weight_path = get_weightPath(args,  False)
        print("Testing: ", weight_path)

        if weight_path == None:
            exit("No Model Saved")

        net = VisDynamicsModel.load_from_checkpoint(weight_path, model=model, **args)
        net.eval()
    
    else:
        net = VisDynamicsModel(model=model, **args)
    
    return net

def prepare_components(args, is_test):

    seed_everything(args.seed)
    
    trainer = prepare_Trainer(args)
    dm = prepare_DataModule(args)
    net = prepare_Model(args, is_test)

    net.example_input_array = torch.rand(1, 3, 128, 256)

    return trainer, dm, net

def train(args):

    trainer, dm, net = prepare_components(args, False)

    trainer.fit(net, dm, ckpt_path=get_weightPath(args, last=True))

def test(args, test_mode="default"):

    trainer, dm, net = prepare_components(args, True)

    net.test_mode = test_mode
    net.pred_log_name = 'predictions'
    net.var_log_name = 'variables'
    net.task_log_dir = 'tasks'

    result = trainer.test(net, dm)
    print(result)
    model_name = create_name(args)
    save_path = os.path.join(net.output_dir, args.dataset, 'tasks', model_name, "test_result")
    mkdir(save_path)
    np.save(os.path.join(save_path, 'results.npy'), result)

    pred(net, args)

# collect variables from training data
def test_on_train_data(args, test_mode="default"):

    trainer, dm, net = prepare_components(args, True)

    net.test_mode = test_mode
    net.pred_log_name = 'predictions_train'
    net.var_log_name = 'variables_train'
    net.task_log_dir = 'tasks_train'
    dm.shuffle = False
    dm.test_dataloader = dm.train_dataloader

    trainer.test(net, dm)

# collect variables from validation data
def test_on_val_data(args, test_mode="default"):

    trainer, dm, net = prepare_components(args, True)

    net.test_mode = test_mode
    net.pred_log_name = 'predictions_val'
    net.var_log_name = 'variables_val'
    net.task_log_dir = 'tasks_val'
    dm.test_dataloader = dm.val_dataloader

    trainer.test(net, dm)

def run_pred(args):
    trainer, dm, net = prepare_components(args, True)

    pred(net, args)



def main():

    parser = argparse.ArgumentParser(description='Neural State Variable training')

    # Mode for script
    parser.add_argument('-mode', help='train or test',
                    type=str, required=True)
    parser.add_argument('-config', help='config file path',
                    type=str, required=True)
    parser.add_argument('-port', help='port number',
                    type=int, required=False, default=8002)

    script_args = parser.parse_args()

    cfg = load_config(filepath=script_args.config)
    pprint.pprint(cfg)

    args = munchify(cfg)

    if script_args.mode ==  "train":
        return train(args)
    elif script_args.mode == "pred":
        return run_pred(args)
    elif script_args.mode == "test":
        return test(args)
    elif script_args.mode ==  "test_all":
        test_on_train_data(args, test_mode="save_train_data")
        test_on_val_data(args, test_mode="save_train_data")
        return test(args)
    elif script_args.mode ==  "show":
        args.port = script_args.port
        return show_nsv(args)


if __name__ == '__main__':
    
    main()