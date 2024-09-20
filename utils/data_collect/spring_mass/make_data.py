import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.integrate import solve_ivp

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def engine(rng, num_frm, fps=60):
    # parameters
    k = 80
    dt = 1.0 / fps
    t_eval = np.arange(num_frm) * dt

    # solve equations of motion
    # y = [x, v]
    f = lambda t, y: [y[1], -k * y[0]]
    initial_state = [rng.uniform(-1, 1), 0]
    sol = solve_ivp(f, [t_eval[0], t_eval[-1]], initial_state, t_eval=t_eval, rtol=1e-6)

    states = sol.y.T
    return states


def preprocess_spring_image():
    bg_color = (215, 205, 192)
    im = Image.open('utils/data_collect/spring_mass/spring.png')
    #print(im.size)
    px = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if (px[i, j][0] > 128) and (px[i, j][1] > 128):
                px[i, j] = bg_color
    im.save('utils/data_collect/spring_mass/spring_processed.png')

def render(x):
    bg_color = (215, 205, 192)
    mass_color = (63, 66, 85)
    im = Image.new('RGB', (800, 800), bg_color)
    spring_im = Image.open('utils/data_collect/spring_mass/spring_processed.png')
    # x in [-1, 1] -> pos in [100, 600]
    pos = int(400 + x * 250)
    spring_im = spring_im.resize((pos-100, 200))
    im.paste(spring_im, (0, 300))
    draw = ImageDraw.Draw(im)
    draw.rectangle((pos-100, 300, pos+100, 500), fill=mass_color) # set position to center of mass

    im = im.resize((128, 128))
    return im


def make_data(data_filepath, num_seq, num_frm, seed=0):
    mkdir(data_filepath)
    rng = np.random.default_rng(seed)
    states = np.zeros((num_seq, num_frm, 2))

    for n in tqdm(range(num_seq)):
        seq_filepath = os.path.join(data_filepath, str(n))
        mkdir(seq_filepath)
        states[n, :, :] = engine(rng, num_frm)
        for k in range(num_frm):
            im = render(states[n, k, 0])
            im.save(os.path.join(seq_filepath, str(k)+'.png'))

    np.save(os.path.join(data_filepath, 'states.npy'), states)


if __name__ == '__main__':
    preprocess_spring_image()
    data_filepath = 'data/spring_mass'
    make_data(data_filepath, num_seq=1200, num_frm=60)