import cv2
import os
import shutil
from tqdm import tqdm
import argparse
import random
import time

## This script is for the purposes of extracting frames from given set of input videos 
## at desired fps.


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--fps', required=True, type=int,
                    help='frames per second at which it is desirable to get output data')
    ap.add_argument('-s', '--seed', default=0, type=int,
                    help='random seed for shuffling')

    args = ap.parse_args()       
    return args


def write_frames(video_path, frame_path, fps, seed):
    # get list of all videos in video_path
    listing = os.listdir(video_path)
    # remove .DS_Store files
    if listing.__contains__('.DS_Store'):
        listing.remove('.DS_Store')
    # random suffle
    random.seed(seed)
    random.shuffle(listing)

    num_videos = len(listing)
    for (idx, filename) in enumerate(listing):
        print(idx+1, 'out of ', num_videos)
        cap = cv2.VideoCapture(os.path.join(video_path, filename))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print('video fps: ', video_fps)
        
        data_path = os.path.join(frame_path, str(idx))
        mkdir(data_path)

        p_frame = 0 # frame count
        skip = round(video_fps / fps)   # sampling frequency
        while(True):
            # read current frame
            ret, frame = cap.read()
            if ret == False:
                break

            # manual crop
            frame = frame[0:620, 371:991]
            # downsize
            frame = cv2.resize(frame, (128, 128), interpolation = cv2.INTER_AREA)

            if p_frame % skip == 0:
                cv2.imwrite(os.path.join(data_path, str(int(p_frame/skip))+'.png'), frame)
            p_frame += 1

        cap.release()
        cv2.destroyAllWindows()
        time.sleep(1.0) # short delay for cleaning purposes


video_path = "./visphysics_data/double_pendulum_raw_videos"
frame_path = "./visphysics_data/double_pendulum_raw_data"
args  = get_arguments()
write_frames(video_path, frame_path, args.fps, args.seed)