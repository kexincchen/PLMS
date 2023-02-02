import sys,getopt
import time
import os.path
from os.path import isfile, join
from os import listdir
import numpy as np
import math
import csv
import re

# the final reward should be the aggregated reward for each window normalized by the completion time.

# this function aggregates the reward.
def calc_aggregate_reward(filename):
    with open(filename, newline = '') as state_file:
        file_reader = csv.reader(state_file)
        aggregate_reward = 0
        window_reward = 0
        prev_row = []
        for row_str in file_reader:
            # row with empty strings means window end
            if ('' in row_str):
                aggregate_reward += window_reward
                window_reward = 0
                continue

            #convert strings to floats in each row
            row = [float(i) for i in row_str]

            #sets previous row
            if prev_row == []:
                prev_row = row
                continue



            #only get reward for the first time an item is selected.
            if (prev_row[2] - row[2]) == 0:
                continue
            if row[2] == 1:
                if row[1] == 0:
                    window_reward += -1
                if row[1] == 1:
                    window_reward += 0.05
                if row[1] == 2:
                    window_reward += 2
            else:
                if row[1] == 0:
                    window_reward += 0.1
                if row[1] == 1:
                    window_reward += 0.05
                if row[1] == 2:
                    window_reward += -2

        #add final window reward as empty row won't be read
        aggregate_reward+=window_reward

        return aggregate_reward

# gets the required files holding state information about the experiment
data_file_path = "../../Data/Participant_Data/RL_CLUSTER"
data_file_suffix = "_states.csv"
state_files = [data_file_path+"/"+f for f in listdir(data_file_path) if (isfile(join(data_file_path,f)) and (data_file_suffix in f))]

# gets the required file for participant performance measures
participant_measure_path = "../../Data/Participant_Data"
participant_measure_filename = "participant_measures.txt"

completion_time = {}

with open(participant_measure_path+"/"+participant_measure_filename) as measure_file:
    lines = measure_file.readlines()
    for line in lines:
        tokens = line.split('\t')
        completion_time[tokens[0]] = tokens[2]

print(completion_time['11'])

#participant data that needs to be removed
to_discard = [5,7,25,29,41,47,62]

for f in state_files:
    #gets pid to pass to reward function
    pid = re.findall(r'\d+',f)

    if(int(pid[0]) in to_discard):
        continue

    print(f"reward for {f} = {calc_aggregate_reward(f)/float(completion_time[pid[0]])}")
