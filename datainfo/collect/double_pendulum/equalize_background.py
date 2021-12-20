import cv2
import os
import shutil
from tqdm import tqdm
import argparse
import numpy as np

'''
This script equalizes backgrounds of all frames with a uniform background.
The background color is user-specified or computed by averaging 
background pixels of sampled frames from the data.
'''


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--channel0', help='b channel')
    ap.add_argument('-g', '--channel1', help='g channel')
    ap.add_argument('-r', '--channel2', help='r channel')

    args = ap.parse_args()                
    return args


raw_filepath = "./visphysics_data/double_pendulum_raw_data"
data_filepath = "./visphysics_data/double_pendulum_background_corrected_data"

num_exps = len(os.listdir(raw_filepath))
bg = np.zeros(3)

args = get_arguments()
if args.channel0 is not None and args.channel1 is not None and args.channel2 is not None:
    # user-specified background color
    bg[0] = args.channel0
    bg[1] = args.channel1
    bg[2] = args.channel2
else:
    # uniform background by averaging sampled pixels
    percent = .05   # proportion of sampled frames
    cnt = 0

    print('Computing uniform background...')
    for p_exp in tqdm(range(num_exps)):
        seq_filepath = os.path.join(raw_filepath, str(p_exp))
        num_frames = len(os.listdir(seq_filepath))
        num_samples = int(percent * num_frames)
        for p_frame in range(num_samples):
            # read each frame
            frame_path = os.path.join(seq_filepath, str(p_frame)+'.png')
            frame = cv2.imread(frame_path)
            # record all background pixels
            for i in range(frame.shape[0]):
                for j in range(frame.shape[1]):
                    # manual threshold
                    if (frame[i,j,2] > 130):
                        bg += frame[i,j]
                        cnt += 1

    # average over all sampled pixels
    bg = bg / cnt
    print('Uniform Background R: ', bg[2], ' G: ', bg[1], ' B: ', bg[0])

# apply the background to all data
print('Applying the background...')
for p_exp in tqdm(range(num_exps)):
    seq_filepath = os.path.join(raw_filepath, str(p_exp))
    data_seq_filepath = os.path.join(data_filepath, str(p_exp))
    mkdir(data_seq_filepath)
    num_frames = len(os.listdir(seq_filepath))

    for p_frame in range(num_frames):
        frame_path = os.path.join(seq_filepath, str(p_frame)+'.png')
        frame = cv2.imread(frame_path)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                # manual threshold
                if (frame[i,j,2] > 130):
                    frame[i,j] = bg
        # write to data
        cv2.imwrite(os.path.join(data_seq_filepath, str(p_frame)+'.png'), frame)