from .angle_estimator import obtain_angle
from .physics_estimator import *

import numpy as np
import os
import sys
import cv2
from tqdm import tqdm


def eval_physics_single_pendulum(frames):

    phys_vars_list =  ['reject', 'theta', 'vel_theta', 'kinetic energy', 'potential energy', 'total energy']
 
    num_frames = len(frames)
    reject = np.zeros(num_frames, dtype=bool)
    theta = np.zeros(num_frames)
    # estimate angles
    marked_images = []
    for p in range(num_frames):
        reject_p, theta_p, img_marked = obtain_angle(frames[p])
        marked_images.append(img_marked)
        if reject_p:
            reject[p] = True
            theta[p] = np.nan
        else:
            reject[p] = False
            theta[p] = theta_p
    # calculate velocities
    vel_theta = np.zeros(num_frames)
    sub_ids = np.ma.clump_unmasked(np.ma.masked_array(theta, reject))
    for ids in sub_ids:
        vel_theta[ids] = calc_velocity(theta[ids].copy())
    vel_theta[reject] = np.nan
    # calculate energies
    kinetic_energy, potential_energy, total_energy = calc_energy(theta, vel_theta)
    # save results
    phys = dict.fromkeys(phys_vars_list)
    phys['reject'] = reject
    phys['theta'] = theta
    phys['vel_theta'] = vel_theta
    phys['kinetic energy'] = kinetic_energy
    phys['potential energy'] = potential_energy
    phys['total energy'] = total_energy




    return phys, marked_images

def eval_phys_data_single_pendulum(data_filepath, num_vids, num_frms, save_path, return_marked=False):

    phys_vars_list =  ['reject', 'theta', 'vel_theta', 'kinetic energy', 'potential energy', 'total energy']

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
        phys_tmp, m_i = eval_physics_single_pendulum(frames)
        marked_images.append(m_i)
        for p_var in phys_vars_list:
            phys[p_var].append(phys_tmp[p_var])
    for p_var in phys_vars_list:
        phys[p_var] = np.array(phys[p_var])

    np.save(save_path, phys)

    if return_marked:
        return marked_images