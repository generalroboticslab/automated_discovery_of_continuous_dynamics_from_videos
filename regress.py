import os
import sys
import time
import pprint
import argparse
import numpy as np
from munch import munchify

from models.nsv_dynamics_model import *
from models.nsv_mlp import *
from models.callbacks import *
from models.data_module import *

from utils.pred import *
from utils.misc import *

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint


def prepare_Trainer(args):

    output_path = os.path.join(os.getcwd(), args.output_dir, args.dataset)
    model_name = create_name(args)

    custom_progress_bar = LitProgressBar()
    
    post_process = RegressEvaluator()

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path +  "/checkpoints/" + model_name,
        verbose=True,
        monitor='val_loss',
        mode='min',
        auto_insert_metric_name=True,
        save_last=True)
    
    cyclic_annealing_callback = CyclicalAnnealingCallback(**args)
    
    callbacks = [custom_progress_bar, post_process, checkpoint_callback, cyclic_annealing_callback]

    logger = pl_loggers.TensorBoardLogger(save_dir=output_path, 
                                        name='logs', 
                                        version=model_name, 
                                        log_graph=True)

    trainer_kwargs = get_validArgs(Trainer, args)

    trainer = Trainer(devices=args.num_gpus,
                      max_epochs=args.epochs,
                      deterministic=True,
                      accelerator='gpu',
                      default_root_dir=output_path,
                      val_check_interval=1.0,
                      callbacks=callbacks,
                      logger=logger,
                      **trainer_kwargs)

    return trainer

def prepare_DataModule(args):

    dm = RegressDataModule(**args)

    return dm

def prepare_Model(args, is_test):

    if 'regressDeeper' in args.model_name:
        model = DeeperNSVMLP(nsv_dim=get_experiment_dim(args.dataset, args.seed), **args)
    else:
        model = NSVMLP(nsv_dim=get_experiment_dim(args.dataset, args.seed), **args)

    if 'smooth' in args.nsv_model_name:
        nsv_model = SmoothNSVAutoencoder.from_model_name(args.nsv_model_name, **args)
    else:
        nsv_model = NSVAutoencoder.from_model_name(args.nsv_model_name, **args)

    
    if is_test:
        
        weight_path = get_weightPath(args, False)
        print("Testing: ", weight_path)

        if weight_path == None:
            exit("No Model Saved")
        
        args.model_name = create_name(args)
        net = NSVDynamicsModel.load_from_checkpoint(weight_path, model=model, nsv_model=nsv_model,  **args)
    
    else:
        args.model_name = create_name(args)
        net = NSVDynamicsModel(model=model, nsv_model=nsv_model, **args)
    
    return net

def prepare_components(args, is_test):

    seed_everything(args.seed)
    
    trainer = prepare_Trainer(args)
    dm = prepare_DataModule(args)
    net = prepare_Model(args, is_test)

    return trainer, dm, net

def train(args):

    tmp = args.model_name
    trainer, dm, net = prepare_components(args, False)

    args.model_name = tmp
    trainer.fit(net, dm, ckpt_path=get_weightPath(args, last=True))

def test(args, percentages=[1,3,5,10]):

    trainer, dm, net = prepare_components(args, True)

    net.test_mode = "default"
    net.var_log_name = 'variables'

    net.percentages = percentages

    result = trainer.test(net, dm)
    print(result)
    save_path = os.path.join(net.output_dir, args.dataset, 'tasks', net.model_name, "test_result")
    print(save_path)
    mkdir(save_path)
    np.save(os.path.join(save_path, 'results.npy'), result)

def test_eq_stability(args, percentages=[1,3,5,10]):

    trainer, dm, net = prepare_components(args, True)

    net.test_mode = "eq_stability"
    net.var_log_name = 'variables'

    net.percentages = percentages

    result = trainer.test(net, dm)

# collect variables from training data
def test_on_train_data(args):

    trainer, dm, net = prepare_components(args, True)

    net.test_mode = "save_train_data"
    net.var_log_name = 'variables_train'
    dm.shuffle = False
    dm.test_dataloader = dm.train_dataloader

    trainer.test(net, dm)

def main():

    parser = argparse.ArgumentParser(description='Neural State Variable Regression training')

    # Mode for script
    parser.add_argument('-mode', help='train or test',
                    type=str, required=True)
    parser.add_argument('-config', help='config file path',
                    type=str, required=True)
    parser.add_argument('-port', help='port number',
                    type=int, required=False, default=8002)
    parser.add_argument('-percentages', help='delta epsilon',
                    type=float, nargs='*', required=False, default=[1,3,5,10, 20])

    script_args = parser.parse_args()

    cfg = load_config(filepath=script_args.config)
    pprint.pprint(cfg)

    args = munchify(cfg)

    if script_args.mode ==  "train":
        return train(args)
    elif script_args.mode ==  "test":
        return test(args, script_args.percentages)
    elif script_args.mode ==  "test_eq":
        return test_eq_stability(args, script_args.percentages)
    elif script_args.mode ==  "test_all":
        tmp = args.model_name
        test_on_train_data(args)
        args.model_name = tmp
        return test(args)
    elif script_args.mode ==  "show":
        args.port = script_args.port
        return show_nsvf(args)


if __name__ == '__main__':
    
    main()