import io
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from neksuite import readnek
from PIL import Image


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def plot_field(x, y, v, vmin, vmax):
    plt.figure()
    plt.xlim([-15, 35])
    plt.ylim([-12.5, 12.5])
    plt.axis('scaled')
    plt.scatter(x, y, c=v, vmin=vmin, vmax=vmax)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    im = Image.open(buf)
    plt.close()
    return im


data_filepath = './test'
phys_filepath = './cylindrical_flow_nekdata'

num_videos = 1200
num_frames = 200
ux_list = np.sqrt(np.linspace(0, 1, num_videos))

for p_vid in range(num_videos):
    ux_ini = ux_list[p_vid]
    print(f'video no. {p_vid}, initial ux={ux_ini}')

    data_vid_filepath = os.path.join(data_filepath, str(p_vid))
    mkdir(data_vid_filepath)
    phys_vid_filepath = os.path.join(phys_filepath, str(p_vid))

    # read all Nek5000 data files
    data_all = []
    for p_frame in range(num_frames):
        nek_filepath = os.path.join(phys_vid_filepath, f'ext_cyl0.f{p_frame}')
        data = readnek(nek_filepath)
        data_all.append(data)

    # extract grid points
    elem_array = data_all[0].elem
    x = np.concatenate([elem.pos[0, 0].flatten() for elem in elem_array])
    y = np.concatenate([elem.pos[1, 0].flatten() for elem in elem_array])

    # extract velocities and plot 
    for p_frame in range(num_frames):
        elem_array = data_all[p_frame].elem
        ux = np.concatenate([elem.vel[0, 0].flatten() for elem in elem_array])
        uy = np.concatenate([elem.vel[1, 0].flatten() for elem in elem_array])
        im1 = plot_field(x, y, ux, -0.6, 1.5)
        im2 = plot_field(x, y, uy, -0.8, 0.8)
        # concatenate and post-process
        im1 = np.array(im1)[50:198, 100:396]
        im2 = np.array(im2)[50:198, 100:396]
        im = np.vstack((im1, im2))[:, :, :3]
        im = Image.fromarray(im).resize((128, 128))
        im.save(os.path.join(data_vid_filepath, f'{p_frame}.png'))