import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.integrate import solve_ivp

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def engine(rng, num_frm, fps=60):
    # parameters
    m = 1.0
    l = 0.5
    g = 9.81
    dt = 1.0 / fps
    t_eval = np.arange(num_frm) * dt

    # solve equations of motion
    # y = [theta, vel_theta]
    f = lambda t, y: [y[1], -3*g/(2*l) * np.sin(y[0])]
    initial_state = [rng.uniform(0, 2*np.pi), rng.uniform(-10, 10)]
    sol = solve_ivp(f, [t_eval[0], t_eval[-1]], initial_state, t_eval=t_eval, rtol=1e-6)
    
    states = sol.y.T
    return states


def render(theta):
    bg_color = (215, 205, 192)
    im = Image.new('RGB', (800, 800), bg_color)
    pendulum_im = Image.open('./pendulum.png')
    im.paste(pendulum_im, (363, 400))
    im = im.rotate(theta*180/np.pi, fillcolor=bg_color)
    im = im.resize((128,128))
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
    data_filepath = '/data/kuang/visphysics_data/single_pendulum'
    make_data(data_filepath, num_seq=1200, num_frm=60)