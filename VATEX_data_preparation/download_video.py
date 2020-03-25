import sys
import json
import os
import tqdm
import subprocess

# get the arguments
if len(sys.argv) <= 2 :
	print("Enter the filename to parse as the first argument of this script")
	print("Enter the output directory to save the downloaded videos as the second argument of this script")
	sys.exit(1)

input_filename = sys.argv[1]
output_dir = sys.argv[2]
log_dir = os.path.join(sys.argv[2], "logs.txt")

print("Logs will be written in {}".format(log_dir))


with open(log_dir, 'a') as l:
	with open(input_filename, 'r') as f:
		json_dict = json.load(f)
		for item in tqdm.tqdm(json_dict):
			video_infos = item['videoID'].split('_')
			start_time = video_infos[-2]
			end_time = video_infos[-1] 
			duration = str(int(end_time) - int(start_time))  # Duration of the video (for FFMPEG to cut X seconds of the video)
			video_id = '_'.join(video_infos[0:-2]) #because IDs can contain '_'  # Name of the video in Youtube
			video_url = 'https://www.youtube.com/watch?v={}'.format(video_id)  # URL of the video to download
			video_output = os.path.join(output_dir, item['videoID']+".mp4")  # last filename
			tmp_video_output = os.path.join(output_dir, "temp_vid.mp4")  # temporary file name

			#print("Downloading the video with id {} from {} to {}".format(video_id, start_time, end_time))
			subprocess.run(["youtube-dl", "--quiet", "--no-warnings", "-f mp4", video_url, "-o", tmp_video_output], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			
			if os.path.isfile(tmp_video_output):  # if the video is found in Youtube
				#print("-----> processing the video seconds")
				subprocess.run(["ffmpeg", "-i", tmp_video_output, "-ss", start_time, "-t", duration, video_output], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
				#print("-----> deleting temporary files")
				subprocess.run(["rm", tmp_video_output], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			else:
				print("{} NOT FOUND !".format(item['videoID']+".mp4"))
				l.write(item['videoID']+"\n")
