from .angle_estimator import obtain_angle
from .physics_estimator import *

import numpy as np
from tqdm import tqdm
import os
import cv2

def eval_physics_double_pendulum(frames):

    phys_vars_list = ['reject', 'theta_1', 'vel_theta_1', 'theta_2', 'vel_theta_2', 'kinetic energy',
'potential energy', 'total energy']
    
    num_frames = len(frames)
    reject = np.zeros(num_frames, dtype=bool)
    theta_1 = np.zeros(num_frames)
    theta_2 = np.zeros(num_frames)
    # estimate angles
    marked_images = []
    for p in range(num_frames):
        reject_p, angles_p, img_marked = obtain_angle(frames[p])
        marked_images.append(img_marked)
        if reject_p:
            reject[p] = True
            theta_1[p] = np.nan
            theta_2[p] = np.nan
        else:
            reject[p] = False
            theta_1[p] = angles_p[0]
            theta_2[p] = angles_p[1]
    # calculate velocities
    vel_theta_1 = np.zeros(num_frames)
    vel_theta_2 = np.zeros(num_frames)
    sub_ids = np.ma.clump_unmasked(np.ma.masked_array(theta_1, reject))
    for ids in sub_ids:
        vel_theta_1[ids] = calc_velocity(theta_1[ids].copy())
        vel_theta_2[ids] = calc_velocity(theta_2[ids].copy())
    vel_theta_1[reject] = np.nan
    vel_theta_2[reject] = np.nan
    # calculate energies
    kinetic_energy, potential_energy, total_energy = calc_energy(theta_1, theta_2, vel_theta_1, vel_theta_2)
    # save results
    phys = dict.fromkeys(phys_vars_list)
    phys['reject'] = reject
    phys['theta_1'] = theta_1
    phys['vel_theta_1'] = vel_theta_1
    phys['theta_2'] = theta_2
    phys['vel_theta_2'] = vel_theta_2
    phys['kinetic energy'] = kinetic_energy
    phys['potential energy'] = potential_energy
    phys['total energy'] = total_energy
    return phys, marked_images

def eval_phys_data_double_pendulum(data_filepath, num_vids, num_frms, save_path, return_marked=False):

    phys_vars_list = ['reject', 'theta_1', 'vel_theta_1', 'theta_2', 'vel_theta_2', 'kinetic energy',
'potential energy', 'total energy']

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
        assert len(frames) == num_frms
        phys_tmp, m_i = eval_physics_double_pendulum(frames)
        marked_images.append(m_i)
        for p_var in phys_vars_list:
            phys[p_var].append(phys_tmp[p_var])
    for p_var in phys_vars_list:
        phys[p_var] = np.array(phys[p_var])

    # remove outliers
    thresh_1 = np.nanpercentile(np.abs(phys['vel_theta_1']), 98)
    thresh_2 = np.nanpercentile(np.abs(phys['vel_theta_2']), 98)
    for n in range(len(num_vids)):
        for p in range(num_frms):
            if (not np.isnan(phys['vel_theta_1'][n, p]) and np.abs(phys['vel_theta_1'][n, p]) >= thresh_1) \
            or (not np.isnan(phys['vel_theta_2'][n, p]) and np.abs(phys['vel_theta_2'][n, p]) >= thresh_2):
                phys['vel_theta_1'][n, p] = np.nan
                phys['vel_theta_2'][n, p] = np.nan
                phys['kinetic energy'][n, p] = np.nan
                phys['total energy'][n, p] = np.nan

    np.save(save_path, phys)

    if return_marked:
        return marked_images