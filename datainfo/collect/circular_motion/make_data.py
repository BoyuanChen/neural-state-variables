import numpy as np
import os
from tqdm import tqdm
import shutil
from PIL import Image, ImageDraw

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


# background and object properties
img_size = (128, 128)
bg_color = 'gray'
obj_size = 8
obj_shape = 'ball'
obj_color ='blue'


def coord2img(x, y):
    img = Image.new('RGB', img_size, bg_color)
    draw = ImageDraw.Draw(img)
    
    pos_x = int(x * img_size[0])
    pos_y = int((1-y) * img_size[1])
    pos = (pos_x-obj_size, pos_y-obj_size, pos_x+obj_size, pos_y+obj_size)
    
    if obj_shape == 'square':
        draw.rectangle(pos, fill=obj_color)
    elif obj_shape == 'ball':
        draw.ellipse(pos, fill=obj_color)
    else:
        assert False
    
    return img


data_filepath = './circular_motion'
num_exp = 1100
num_frames = 60

# data_filepath = './circular_motion_long'
# num_exp = 1
# num_frames = 60*20

center = (0.5, 0.5)
radius = 0.3
np.random.seed(0)

for p_exp in tqdm(range(num_exp)):
    theta = np.random.uniform(0, 2*np.pi)
    theta_d = np.random.choice((-1, 1)) * np.random.uniform(np.pi/30, np.pi/10)

    seq_filepath = os.path.join(data_filepath, str(p_exp))
    mkdir(seq_filepath)

    for p_frame in range(num_frames):
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        img = coord2img(x, y)
        theta += theta_d
        img.save(os.path.join(seq_filepath, str(p_frame)+'.png'))	