import os
import time
import shutil
import numpy as np
from neksuite import readnek


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


usr_filepath = './ext_cyl.usr'
phys_filepath = './cylindrical_flow_nekdata'

num_videos = 1200
num_frames = 200
ux_list = np.sqrt(np.linspace(0, 1, num_videos))

for p_vid in range(num_videos):
    ux_ini = ux_list[p_vid]
    print(f'video no. {p_vid}, initial ux={ux_ini}')

    # modify .usr file
    with open(usr_filepath, 'r') as file:
        lines = file.readlines()
    line_numbers = [91, 102]
    new_lines = [f'      ux={ux_ini}', f'      ux={ux_ini}']
    for line_number, new_line in zip(line_numbers, new_lines):
        lines[line_number - 1] = new_line + '\n'
    with open(usr_filepath, 'w') as file:
        file.writelines(lines)

    # run simulation
    os.system('makenek ext_cyl')
    os.system('nekbmpi ext_cyl 8')
    # wait for enough time to finish simulation
    time.sleep(300)

    phys_vid_filepath = os.path.join(phys_filepath, str(p_vid))
    mkdir(phys_vid_filepath)

    # save all Nek5000 data files and get ux, uy limits
    ux_lims = [1e7, -1e7]
    uy_lims = [1e7, -1e7]

    for p_frame in range(num_frames):
        nek_filepath = f'./ext_cyl0.f{p_frame+1:05d}'
        data = readnek(nek_filepath)
        print(data.time)
        ux_lims[0] = min(ux_lims[0], data.lims.vel[0, 0])
        ux_lims[1] = max(ux_lims[1], data.lims.vel[0, 1])
        uy_lims[0] = min(uy_lims[0], data.lims.vel[1, 0])
        uy_lims[1] = max(uy_lims[1], data.lims.vel[1, 1])
        nek_filepath_new = os.path.join(phys_vid_filepath, f'ext_cyl0.f{p_frame}')
        os.system(f'mv {nek_filepath} {nek_filepath_new}')

    print(ux_lims, uy_lims)