

import os
import shutil
import numpy as np
from tqdm import tqdm



def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

data_folder = '/home/cml/Downloads/visphysics_data'
object_lst = ['air_dancer', 'fire', 'swingstick_magnetic', 'swingstick_non_magnetic']
num_train_vids = [22, 1, 18, 68]
num_test_vids = [5, 1, 5, 17]

new_data_folder=  '/home/cml/Downloads/visphysics_data_new'


for i in tqdm(range(len(object_lst))):
    obj = object_lst[i]
    old_folder = os.path.join(data_folder, obj)
    new_folder = os.path.join(new_data_folder, obj)
    # train
    num_train_frames = len(os.listdir(os.path.join(old_folder, 'raw_data_train')))
    ids = np.array(list(range(num_train_frames)))
    ids = np.split(ids, num_train_vids[i])
    for j in tqdm(range(num_train_vids[i])):
        new_vid_folder = os.path.join(new_folder, 'train', str(j))
        mkdir(new_vid_folder)
        this_ids = ids[j]
        k = 0
        for p_idx in this_ids:
            src = os.path.join(old_folder, 'raw_data_train', str(p_idx) + '.jpg')
            dest = os.path.join(new_vid_folder, str(k) + '.jpg')
            shutil.copy(src, dest)
            k = k + 1
    # test
    num_test_frames = len(os.listdir(os.path.join(old_folder, 'raw_data_test')))
    ids = np.array(list(range(num_test_frames)))
    ids = np.split(ids, num_test_vids[i])
    for j in tqdm(range(num_test_vids[i])):
        new_vid_folder = os.path.join(new_folder, 'test', str(j))
        mkdir(new_vid_folder)
        this_ids = ids[j]
        k = 0
        for p_idx in this_ids:
            src = os.path.join(old_folder, 'raw_data_test', str(p_idx) + '.png')
            dest = os.path.join(new_vid_folder, str(k) + '.png')
            shutil.copy(src, dest)
            k = k + 1



