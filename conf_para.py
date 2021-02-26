import os
from time import time
from glob import glob
from random import choice

import numpy as np
from pandas import DataFrame

from eegnb import generate_save_fn
from eegnb.stimuli import CONFUSION_VID

__title__ = "Confusion Paradigm"

class Video(object):
    def __init__(self,path):
        self.path = path

    def play(self):
        from os import startfile
        startfile(self.path)

class Movie_MP4(Video):
    type = "MP4"



def present(duration=120, eeg=None, save_fn=None):
    vid_targets = list(glob(os.path.join(CONFUSION_VID, "*.mp4")))
    num_trials = len(vid_targets)
    record_duration = np.float32(duration)

    answers = DataFrame()

    # Randomize order that videos are played in
    vid_num = np.arange(num_trials) 
    np.random.shuffle(vid_num)

    vid_targets = list(glob(os.path.join(CONFUSION_VID, "*.mp4")))

    difficulties = []
    easy_topics = []
    hard_topics = []
    vid_names = []


    # Show instructions
    show_instructions(duration=duration)

    # start the EEG stream, will delay 5 seconds to let signal settle
    if eeg:
        if save_fn is None:  # If no save_fn passed, generate a new unnamed save file
            save_fn = generate_save_fn(eeg.device_name, "visual_p300", "unnamed")
            print(
                f"No path for a save file was passed to the experiment. Saving data to {save_fn}"
            )
        eeg.start(save_fn, duration=record_duration)

    # Iterate through the events
    start = time()

    for i in range(num_trials):
        vid_index = vid_num[i]
        vid_path = vid_targets[vid_index]
        vid_name = os.path.basename(vid_path)
        movie = Movie_MP4(vid_path)

        if input("Press enter to play, anything else to exit") == '':
            movie.play()
        timestamp = time()
        difficulty = input("On a scale of 1-5, how confusing was this video?: ")
        difficulties.append(difficulty)
        easy_topic = input("Which topics (if any) from this video \"clicked\" for you? ")
        easy_topics.append(easy_topic)
        hard_topic = input("Which topics (if any) from this video were confusing for you? ")
        hard_topics.append(hard_topic)
        vid_names.append(vid_name)

        if eeg:
            timestamp = time()
            if eeg.backend == "muselsl":
                marker = vid_path #Alter to use .csv file to associate .mp4 filename with confusion classication
            else:
                marker = vid_path #Alter to use .csv file to associate .mp4 filename with confusion classication
            eeg.push_sample(marker=marker, timestamp=timestamp)

    if eeg:
        eeg.stop()
        answers["Video_Names"] = vid_names
        answers["Self_Reported_Difficulty"] = difficulties
        answers["Hard_Topics"] = hard_topics
        answers["Easy_Topics"] = easy_topics
        answers.to_csv(os.path.join(r"C:\Users\alexp\Desktop\School\4th_Year\MactionPotential\eeg-notebooks\eegnb\experiments\visual_videos", "Output.csv"), sep=",")


def show_instructions(duration):

    instruction_text = """
    Welcome to the experiment! 
 
    Stay still, focus on the centre of the screen, and try not to blink. 

    This block will run for %s seconds.

    Press spacebar to continue. 
    
    """
    instruction_text = instruction_text % duration
    print(instruction_text)


