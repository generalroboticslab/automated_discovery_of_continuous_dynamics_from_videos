import random 
import os
from misc import mkpath
import json
import argparse


def create_random_data_splits(seed, data_filepath, object_name, num_vids, ratio=0.8):
    random.seed(seed)
    seq_dict = {}
    vid_id_lst = list(range(num_vids))
    random.shuffle(vid_id_lst)
    # test
    start = int(num_vids * (ratio + (1 - ratio) / 2))
    seq_dict['test'] = vid_id_lst[start:]
    # val
    start = int(num_vids * ratio)
    end = int(num_vids * (ratio + (1 - ratio) / 2))
    seq_dict['val'] = vid_id_lst[start:end]
    # train
    seq_dict['train'] = vid_id_lst[:int(num_vids * ratio)]
    # mkdir first
    obj_filepath = os.path.join(data_filepath, object_name, 'datainfo')
    mkpath(obj_filepath)
    with open(os.path.join(obj_filepath, f'data_split_dict_{seed}.json'), 'w') as file:
        json.dump(seq_dict, file, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Neural State Variable training')

    # Mode for script
    parser.add_argument('-dataset', help='dataset',
                    type=str, required=True)
    
    script_args = parser.parse_args()

    data_filepath = 'data'
    object_name = script_args.dataset
    num_vids = 1099 if 'double_pendulum' in object_name  else 1200
    seeds = [1,3,4] if object_name == 'spring_mass' else [1,2,3]
    for seed in seeds:
        create_random_data_splits(seed, data_filepath, object_name, num_vids)