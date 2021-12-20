## Double Pendulum Data Collection

Suppose the raw videos recording the double pendulum motion are stored in ```./visphysics_data/double_pendulum_raw_videos``` in the current folder.
Run the following commands.
The collected data are saved in ```./visphysics_data/double_pendulum```.

1. python convert_video.py -f 60 (convert each video to a sequence of images with sampling frequency 60fps)
2. python equalize_background.py -r 215 -g 205 -b 192 (equalize the background color)
3. python split_data.py (split each sequence 
into subsequences with fixed length 60)