import os
import sys
import json
import jieba

# get the arguments
if len(sys.argv) <= 2 :
	print("Enter the filename as the first argument of this script and the output as the second argument")
	sys.exit(1)


input_filename = sys.argv[1]
output_filename = sys.argv[2]



if("test" in input_filename): #if file just export the video
	video_file = output_filename+".video"
	en_file = output_filename+".en"
	with open(video_file, 'w') as vid_f:
		with open(en_file, 'w') as en_f:
			with open(input_filename, 'r') as original_dataset:
				json_dict = json.load(original_dataset)
				for item in json_dict:
					video_id = item['videoID']
					en_txt = item['enCap'][5::]
					[vid_f.write(video_id+"\n") for _ in range(5)]
					[en_f.write(txt+"\n") for txt in en_txt]

else: #train + val
	en_file = output_filename+".en"
	ch_file = output_filename+".ch"
	video_file = output_filename+".video"

	i = 0
	with open(video_file, 'w') as vid_f:
		with open(en_file, 'w') as en_f:
			with open(ch_file, 'w') as ch_f:
				with open(input_filename, 'r') as original_dataset:
					json_dict = json.load(original_dataset)
					for item in json_dict:
						video_id = item['videoID']
						#only select last 5 captions because it's the parallel translation
						en_txt = item['enCap'][5::] 
						ch_txt = item['chCap'][5::]

						for i in range(len(ch_txt)): # segmentation of the chinese text
							ch_txt[i] = " ".join(jieba.cut(ch_txt[i], cut_all=False))
						
						[vid_f.write(video_id+"\n") for _ in range(5)]
						[en_f.write(txt+"\n") for txt in en_txt]
						[ch_f.write(txt+"\n") for txt in ch_txt]

