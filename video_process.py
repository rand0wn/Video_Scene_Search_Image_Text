"""Class for training video for search and performing other operations."""

import numpy as np
import cv2
import pandas as pd
import run_inference

# Video Data
vd_data = './vd_data'
vd_df = pd.read_csv(vd_data+'/vd_data.csv')
len_vd_df = vd_df.shape[0]


class Video(object):

    def __init__(self,
                 filenames,
                 frame_frequency,
                 audio_or_sub):

        # Class Parameters
        self.filenames = filenames.split(',')   # Video Files
        self.frame_frequency = frame_frequency  # Number of frames to consider
        self.audio_or_sub = audio_or_sub        # With audio or subs

    # Train Videos and Add Data to DataFrame
    def train_videos(self):
        global vd_df
        global len_vd_df
        for filename in self.filenames:
            # init for video
            vd_cap = cv2.VideoCapture(filename)
            frame_count = 0
            video_name = filename.split('/')[len(filename.split('/'))-1]

            while vd_cap.isOpened():
                ret, frame = vd_cap.read()
                # Check for video
                if ret:
                    frame_count = frame_count + 1

                    # For every 5 secs, 24 fps
                    if frame_count % 120 == 0:
                        frame_img_name = video_name + '_' + str(frame_count/24) + '.jpg'
                        frame_img_loc = vd_data + '/frames/' + frame_img_name
                        cv2.imwrite(frame_img_loc, frame)

                        vd_df = vd_df.append({'frame': frame_img_name, 'video': video_name, 'time': frame_count/24, 'prob': 0, 'caps': 0, 'words': 0, 'tags': 0, 'subs': 0}, ignore_index=True)
                else:
                    # List of All Video Frames
                    frame_list = [vd_data+'/frames/' + x for x in list(vd_df['frame'][len_vd_df:,])]

                    # Store Show and Tell Model Captions and Prob
                    file_input = ['models/model2.ckpt-2000000', 'models/word_counts.txt', ",".join(frame_list)]
                    prob, cap = run_inference.img_captions(file_input)
                    vd_df.iloc[len_vd_df:, vd_df.columns.get_loc("prob")] = prob
                    vd_df.iloc[len_vd_df:, vd_df.columns.get_loc("caps")] = cap

                    # Update Final Changes to CSV
                    vd_df.to_csv(vd_data + '/vd_data.csv', index=False)
                    len_vd_df = vd_df.shape[0]  # Update Length
                    break

# Train Videos
v = Video(filenames='', frame_frequency=0, audio_or_sub=0)

v.train_videos()