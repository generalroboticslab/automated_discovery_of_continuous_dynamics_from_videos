import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch

ux_range = [-0.6, 1.5]
uy_range = [-0.8, 0.8]
rho = 0.00024303899906795914

colors = plt.cm.viridis.colors
L = len(colors)

def find_value_colormap(img, c2idx):
    
    return c2idx(torch.from_numpy(img).float().cuda()).detach().cpu().numpy()

def get_vectorField(img, c2idx):

    d = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0

    img_ux = d[:64,:,:]
    img_uy = d[64:,:,:]

    ux = np.zeros(img_ux.shape[:2])
    uy = np.zeros(img_uy.shape[:2])

    ux_float = find_value_colormap(img_ux, c2idx)
    if np.any(ux_float > 1) or np.any(ux_float < 0):
        return True, None
    uy_float = find_value_colormap(img_uy, c2idx)
    if np.any(uy_float > 1) or np.any(uy_float < 0):
        return True, None

    ux = ux_range[0] + (ux_range[1]-ux_range[0]) * ux_float
    uy = uy_range[0] + (uy_range[1]-uy_range[0]) * uy_float

    #print(ux.shape)
    #print(img_uy)

    vf = np.stack((ux,uy), axis=-1)

    return False, vf

def estimate_energy(v):

    e = np.sum(0.5*rho*v**2)

    #print(e)
    
    return e

def obtain_energy(img, c2idx, longer=False):
    img_marked = img.copy()

    rej, v = get_vectorField(img, c2idx)
    if not rej:
        energy = estimate_energy(v)
        # mark the estimated energy 
        if longer:
            cv2.putText(img_marked, f'{energy:.4f}', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        else:
            cv2.putText(img_marked, f'{energy:.2f}', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
    else:
        # mark the rejection
        cv2.putText(img_marked, 'Reject', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        energy = np.nan

    return rej, energy, img_marked