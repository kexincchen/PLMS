import sys
from os.path import isfile, join
from os import listdir
import csv
import re
import matplotlib.pyplot as plt
import numpy as np

# the final reward should be the aggregated reward for each window normalized by the completion time.
# this function aggregates the reward.
def calc_aggregate_reward(filename, num_steps = -1):
    with open(filename, newline = '') as state_file:
        file_reader = csv.reader(state_file)

        aggregate_reward = 0
        step_reward = 0
        prev_row = []
        curr_num_steps = 0
        count_rows = 0

        rewards_steps_dict = {}

        for row_str in file_reader:
            count_rows += 1

            # row with empty strings means window end
            if ('' in row_str):
                aggregate_reward += step_reward
                step_reward = 0

                curr_num_steps += 1
                if (curr_num_steps != 0 and curr_num_steps%num_steps == 0):
                    rewards_steps_dict[curr_num_steps] = aggregate_reward

                continue

            #convert strings to floats in each row
            row = [float(i) for i in row_str]

            #sets previous row
            if prev_row == []:
                prev_row = row
                continue

            #for selected
            if row[2] == 1:
                # only get reward for the first time an item is selected.
                if (prev_row[2] - row[2]) == 0:
                    continue
                if row[1] == 0:
                    step_reward += -1
                if row[1] == 1:
                    step_reward += 0.05
                if row[1] == 2:
                    step_reward += 2
            # contrary to previous case, get reward/punishment for every step items are not selected
            else:
                if row[1] == 0:
                    step_reward += 0.1
                if row[1] == 1:
                    step_reward += 0.05
                if row[1] == 2:
                    step_reward += -2

        #add final window reward as empty row won't be read
        aggregate_reward += step_reward

        curr_num_steps += 1
        rewards_steps_dict[curr_num_steps] = aggregate_reward

        return aggregate_reward, count_rows+1, rewards_steps_dict


# gets the required files holding state information about the experiment
data_file_path = "../../Data/Participant_Data/RL_CLUSTER"
data_file_suffix = "_states.csv"
state_files = [data_file_path+"/"+f for f in listdir(data_file_path) if (isfile(join(data_file_path,f)) and (data_file_suffix in f))]

# gets the required file for participant performance measures
participant_measure_path = "../../Data/Participant_Data"
participant_measure_filename = "participant_measures.txt"


def main(argv):
    completion_time = {}

    with open(participant_measure_path+"/"+participant_measure_filename) as measure_file:
        lines = measure_file.readlines()
        for line in lines:
            tokens = line.split('\t')
            completion_time[tokens[0]] = tokens[2]


    #participant data that needs to be removed
    to_discard = [5,7,25,29,41,47,62]

    for f in state_files:
        #gets pid to pass to reward function
        pid = re.findall(r'\d+',f)

        if(int(pid[0]) in to_discard):
            continue

        agg_reward, count_rows, rewards_step_dict = calc_aggregate_reward(f,int(argv[0]))
        steps = count_rows/61 # there are 60 post-its and 1 blank for each step

        print(f"\nrows in {f} = {count_rows}")
        print(f"reward per second {f} = {agg_reward/float(completion_time[pid[0]])} \nreward per step {f} = {agg_reward/steps}")

        if(pid[0] == '61'):
            print(rewards_step_dict)
            plt.plot(rewards_step_dict.keys(),rewards_step_dict.values())
            plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])