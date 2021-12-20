import os
import shutil
from tqdm import tqdm
import argparse

'''
This script splits each sequence, which corresponds to a single experiment,
into subsequences with fixed length.
'''


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--len', type=int, default=60, help='length of each subsequence')

    args = ap.parse_args()                
    return args


src_filepath = "./visphysics_data/double_pendulum_background_corrected_data"
dst_filepath = "./visphysics_data/double_pendulum"

args = get_arguments()

num_exps = len(os.listdir(src_filepath))
subseq_cnt = 0
frame_cnt = 0
mkdir(os.path.join(dst_filepath, str(subseq_cnt)))

for p_exp in tqdm(range(num_exps)):
    src_seq_filepath = os.path.join(src_filepath, str(p_exp))
    num_frames = len(os.listdir(src_seq_filepath))
    # ignore frames at the end of the sequence
    num_frames = int(num_frames/args.len) * args.len

    for p_frame in range(num_frames):
        src_path = os.path.join(src_seq_filepath, str(p_frame)+'.png')
        dst_path = os.path.join(dst_filepath, str(subseq_cnt), str(frame_cnt)+'.png')
        shutil.copyfile(src_path, dst_path)

        frame_cnt += 1
        if frame_cnt == args.len:
            subseq_cnt += 1
            frame_cnt = 0
            mkdir(os.path.join(dst_filepath, str(subseq_cnt)))

# remove the last empty folder
shutil.rmtree(os.path.join(dst_filepath, str(subseq_cnt)))