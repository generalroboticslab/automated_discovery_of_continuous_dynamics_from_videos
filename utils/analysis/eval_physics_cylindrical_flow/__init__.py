import numpy as np
import os
import sys
import cv2
from tqdm import tqdm
from .energy_estimator import obtain_energy


def eval_phys_cylindrical_flow(frames, c2idx, longer=False):

    phys_vars_list =  ['reject', 'total energy']

    num_frames = len(frames)
    reject = np.zeros(num_frames, dtype=bool)
    energy = np.zeros(num_frames)

    marked_images = []

    for p in range(num_frames):

        reject_p, e_p, img_marked = obtain_energy(frames[p], c2idx, longer)
        marked_images.append(img_marked)
        if reject_p:
            reject[p] = True
            energy[p] = np.nan
        else:
            reject[p] = False
            energy[p] = e_p
    
    phys = dict.fromkeys(phys_vars_list)
    phys['reject'] = reject
    phys['total energy'] = energy
        
    return phys, marked_images


def eval_phys_data_cylindrical_flow(data_filepath, num_vids, num_frms, save_path, c2idx, return_marked=False):

    phys_vars_list =  ['reject', 'total energy']
    phys = {p_var:[] for p_var in phys_vars_list}

    marked_images = []

    for n in tqdm(num_vids):
        seq_filepath = os.path.join(data_filepath, str(n))
        images = [img for img in os.listdir(seq_filepath)
            if img.endswith(".png")]

        images.sort(key = lambda x: int(x[:-4]))
        frames = []
        for p in images:
            frame_p = cv2.imread(os.path.join(seq_filepath, p))
            frames.append(frame_p)
        phys_tmp, m_i = eval_phys_cylindrical_flow(frames, c2idx, longer='filtered' not in save_path)
        marked_images.append(m_i)
        for p_var in phys_vars_list:
            phys[p_var].append(phys_tmp[p_var])
    for p_var in phys_vars_list:
        phys[p_var] = np.array(phys[p_var])
    
    np.save(save_path, phys)

    if return_marked:
        return marked_images