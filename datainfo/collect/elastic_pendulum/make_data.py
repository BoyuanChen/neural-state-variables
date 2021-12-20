import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy import linalg
from scipy.integrate import solve_ivp

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def engine(rng, num_frm, fps=60):
    # physical parameters
    L_0 = 0.205  # elastic pendulum unstretched length (m)
    L = 0.179    # rigid pendulum rod length (m)
    w = 0.038    # rigid pendulum rod width (m)
    m = 0.110    # rigid pendulum mass (kg)
    g = 9.81     # gravitational acceleration (m/s^2)
    k = 40.0     # elastic constant (kg/s^2)

    dt = 1.0 / fps
    t_eval = np.arange(num_frm) * dt
    
    # solve equations of motion
    # y = [theta_1, theta_2, z, vel_theta_1, vel_theta_2, vel_z]
    def f(t, y):
        La = L_0 + y[2]
        Lb = 0.5 * L
        I = (1. / 12.) * m * (w ** 2 + L ** 2)
        Jc = L * np.cos(y[0] - y[1])
        Js = L * np.sin(y[0] - y[1])
        A = np.array([[La**2, 0.5*La*Jc, 0],
                      [0.5*La*Jc, Lb**2+I/m, 0.5*Js],
                      [0, 0.5*Js, 1]])
        b = np.zeros(3)
        b[0] = -0.5*La*Js*y[4]**2 - 2*La*y[3]*y[5] - g*La*np.sin(y[0])
        b[1] = 0.5*La*Js*y[3]**2 - Jc*y[3]*y[5] - g*Lb*np.sin(y[1])
        b[2] = La*y[3]**2 + 0.5*Jc*y[4]**2 + g*np.cos(y[0]) - (k/m)*y[2]
        sol = linalg.solve(A, b)
        return [y[3], y[4], y[5], sol[0], sol[1], sol[2]]

    # run until the drawn pendulums are inside the image
    rej = True
    while rej:
        initial_state = [rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi), rng.uniform(-0.04, 0.04), rng.uniform(-6, 6), rng.uniform(-10, 10), 0]
        sol = solve_ivp(f, [t_eval[0], t_eval[-1]], initial_state, t_eval=t_eval, rtol=1e-6)
        states = sol.y.T
        lim_x = np.abs((L_0+states[:, 2])*np.sin(states[:, 0]) + L*np.sin(states[:, 1])) - (L_0 + L)
        lim_y = np.abs((L_0+states[:, 2])*np.cos(states[:, 0]) + L*np.cos(states[:, 1])) - (L_0 + L)
        rej = (np.max(lim_x) > 0.16) | (np.max(lim_y) > 0.16) | (np.min(states[:, 2]) < -0.08)
    
    return states


def draw_rect(im, col, top_x, top_y, w, h, theta):
    x1 = top_x - w * np.cos(theta) / 2
    y1 = top_y + w * np.sin(theta) / 2
    x2 = x1 + w * np.cos(theta)
    y2 = y1 - w * np.sin(theta)
    x3 = x2 + h * np.sin(theta)
    y3 = y2 + h * np.cos(theta)
    x4 = x3 - w * np.cos(theta)
    y4 = y3 + w * np.sin(theta)
    pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    draw = ImageDraw.Draw(im)
    draw.polygon(pts, fill=col)


def render(theta_1, theta_2, z):
    bg_color = (215, 205, 192)
    pd1_color = (63, 66, 85)
    pd2_color = (17, 93, 234)

    im = Image.new('RGB', (1600, 1600), bg_color)
    center = (800, 800)

    w1, w2, h1, h2 = 90, 90, 300, 250
    L_0 = 0.205
    h1 *= (1 + z / L_0)
    pd1_end_x = center[0] + (h1-35) * np.sin(theta_1)
    pd1_end_y = center[1] + (h1-35) * np.cos(theta_1)

    # pd1 may hide pd2
    draw_rect(im, pd2_color, pd1_end_x, pd1_end_y, w2, h2, theta_2)
    draw_rect(im, pd1_color, center[0], center[1], w1, h1, theta_1)

    im = im.resize((128, 128))
    return im


def make_data(data_filepath, num_seq, num_frm, seed=0):
    mkdir(data_filepath)
    rng = np.random.default_rng(seed)
    states = np.zeros((num_seq, num_frm, 6))

    for n in tqdm(range(num_seq)):
        seq_filepath = os.path.join(data_filepath, str(n))
        mkdir(seq_filepath)
        states[n, :, :] = engine(rng, num_frm)
        for k in range(num_frm):
            im = render(states[n, k, 0], states[n, k, 1], states[n, k, 2])
            im.save(os.path.join(seq_filepath, str(k)+'.png'))

    np.save(os.path.join(data_filepath, 'states.npy'), states)


if __name__ == '__main__':
    data_filepath = '/data/kuang/visphysics_data/elastic_pendulum'
    make_data(data_filepath, num_seq=1200, num_frm=60)