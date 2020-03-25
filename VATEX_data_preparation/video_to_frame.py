import cv2
import json
import tqdm
import sys
import numpy as np
import os
from time import sleep

# get the arguments
if len(sys.argv) <= 3 :
	print("Enter the filename to parse as the first argument of this script")
	print("Enter the folder where the video are located as the second argument of this script")
	print("Enter the output filename as the third argument of this script")
	sys.exit(1)

filename = sys.argv[1]
video_path = sys.argv[2]
output_path = sys.argv[3]

with open(filename, 'r') as f:
	json_dict = json.load(f)
	frames_per_video = []
	for item in tqdm.tqdm(json_dict):
		video_id = "_".join(item['videoID'])+".mp4"
		
		cap = cv2.VideoCapture(os.path.join(video_path, video_id))
		video_frames = []
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret:
			    video_frames.append(frame)
			else:
				cap.release()
				cv2.destroyAllWindows()
		print(np.array(video_frames).shape)
		sys.exit
		frames_per_video.append(video_frames)
		break
	frames_per_video = np.array(frames_per_video)
	print(frames_per_video.shape)

		
