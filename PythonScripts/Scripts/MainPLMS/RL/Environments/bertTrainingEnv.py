from collections import OrderedDict
from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiDiscrete, Tuple
import numpy as np, numpy.random
import random
import copy
import math
import sys

sys.path.append("../..")

from TextProcessor.textProcessor import TextProcessor

##################################################################
#generates topic probabilities for each episode
##################################################################
class BertTrainingEnv(Env):
    def __init__(self, number_clues, nn_input_rows, embedding_size, number_states, filenames, textProcessor, dwellEpsilon, maxPostPerStep): # we pass the files as topic probs change. Only useful for training

        self.number_clues = number_clues
        self.nn_input_rows = nn_input_rows
        self.embedding_size = embedding_size
        self.number_states = number_states
        self.dwellEpsilon = dwellEpsilon
        self.maxPostPerStep = maxPostPerStep

        self.filenames = filenames
        self.textProcessor = textProcessor
        postIts = self.textProcessor.postItsFromFile_bert(self.filenames)
        self.start_state = BertTrainingEnv.getBox(postIts,self.embedding_size, self.nn_input_rows)

        ##################################################################
        #BOX
        ##################################################################

        # #number of clues x (embedding size) + 1(states) + 1 (isSelected) + 1 (postIt eye dwell time ratio) + 1 (saccadeIn Ratio)
        # obs_high = np.ones(shape = (self.number_clues,self.embedding_size+4), dtype = np.float32)
        # obs_high[:,self.embedding_size] = np.float32(2)
        # obs_low = np.zeros(shape = (self.number_clues,self.embedding_size+4), dtype = np.float32)
        # self.observation_space = Box(low=obs_low, high = obs_high, dtype = np.float32)

        obs_high = np.ones(shape = (self.nn_input_rows,self.embedding_size+4), dtype = np.float32)
        obs_high[:,self.embedding_size] = np.float32(2)
        obs_low = np.zeros(shape = (self.nn_input_rows,self.embedding_size+4), dtype = np.float32)
        self.observation_space = Box(low=obs_low, high = obs_high, dtype = np.float32)

        ##################################################################

        # all three are the same
        # self.action_space = Box(low = np.int32(0), high = np.int32(2), shape = (self.number_clues,), dtype = np.int32)
        # # self.action_space = Box(low = np.zeros(self.number_clues), high = np.ones(self.number_clues)*2, dtype = np.int32)
        # # self.action_space = MultiDiscrete(np.ones(self.number_clues)*3)

        # self.action_space = Box(low = np.int32(0), high = np.int32(2), shape = (self.nn_input_rows,), dtype = np.int32)
        # # self.action_space = Box(low = np.zeros(self.number_clues), high = np.ones(self.nn_input_rows)*2, dtype = np.int32)
        # # self.action_space = MultiDiscrete(np.ones(self.nn_input_rows)*3)


        # for SOFTMAX
        # self.action_space = Box(low = np.int32(0), high = np.int32(1), shape = (self.number_clues,self.number_states), dtype = np.float32)

        self.action_space = Box(low = np.int32(0), high = np.int32(1), shape = (self.nn_input_rows,self.number_states), dtype = np.float32)
        ##################################################################

        #setting state to start_state
        self.state = self.start_state
        self.previous_state = copy.deepcopy(self.state)

        #length
        self.train_length = 360

    ##################################################################
    # WITH BOX SPACE
    # used when observation_space is represented by gy.spaces.Box
    ##################################################################
    # state[:,0:number_topics] -> topic_probs
    # state[:,embedding_size] -> states
    # state[:,embedding_size+1] -> isSelected
    # state[:,embedding_size+2] -> dwellTimeRatio
    # state[:,embedding_size+3] -> saccadeInRatio
    def step(self, action):
        #set previous state
        self.previous_state = copy.deepcopy(self.state)

        #############################################
        # used when softmax and Discretization layers are not used
        #############################################
        # # lower_bound_action = 0
        # # upper_bound_action = 2
        # #
        # # # round the contiuous output presented by the actor model to a discrete space
        # # # also convert to int32 to match our action_space
        # # # print(action)
        # # action = np.clip(action, lower_bound_action, upper_bound_action)
        # # action = np.round(action).astype(np.int32)
        #
        # # Since the above method with output_layer * 2 seems to make very minor adjustments
        # # there does not seem to be enough movement to show any change in the HoloLens post it notes
        # # the following works with output_layer * 3 in the ddpgLearning actor method
        # lower_bound_action = 0
        # upper_bound_action = 2.9
        # action = np.clip(action, lower_bound_action, upper_bound_action)
        # action = np.floor(action).astype(np.int32)
        # assert self.action_space.contains(action)


        #############################################
        # if using softmax
        # DOES NOT WORK
        #############################################
        # this section should give me the shape of the action_space for the SOFTMAX variant
        lower_bound_action = 0
        upper_bound_action = 1
        action = np.clip(action, lower_bound_action, upper_bound_action)
        # action = action.reshape(self.number_clues, self.number_states)
        action = action.reshape(self.nn_input_rows, self.number_states)

        assert self.action_space.contains(action)


        #turn softmax to required actions using argmax
        action = np.argmax(action, axis = -1)

        # print(action)

        self.state[:,self.embedding_size] = action
        self.train_length -= 1

        #send action to HoloLens

        #simulate user behaviour
        #blocking wait for new state from HoloLens

        # use k-means type clustering to determine if something is closer to a selected or non-selected topic

        # calculate average topic probability of
        # Selected topics
        # and topics that have been looked at for a long enough time
        avg_selected = np.zeros(shape = (self.embedding_size))
        selected = 0

        # keeps track of all the selected notes so that they are not considered in non_selected
        selectedIndices = []

        for i in range(0,self.number_clues):
            if(
                (self.state[i,(self.embedding_size+1)] == 1) # checks if this has been posted
                # scale dwell time based on the state -> if highlighted = more dwell
                # or ((self.state[i,self.embedding_size+2] * (self.state[i,self.embedding_size]+1)) > (random.uniform(0.70, 0.95)*self.number_states))
                or ((self.state[i,self.embedding_size+2] * (self.state[i,self.embedding_size]+1)) > (random.uniform(0.50, 0.75)*self.number_states))
                or (self.state[i,self.embedding_size+3] > (random.uniform(0.50, 0.90))) # 0.90 and not 0.75 as with dwell time because eye tracking is not very accurate and flickering in and out of a note was seen occassionaly, therefore we account for this with larger value of 0.9
              ):
                selected = selected+1
                avg_selected = avg_selected + self.state[i,0:self.embedding_size]
                selectedIndices.append(i)
        if(selected != 0):
            avg_selected = avg_selected/selected

        # calculate average topic probability of
        # non-selected topics
        # and topics that have been looked at for a long enough time
        avg_non_selected = np.zeros(shape = (self.embedding_size))
        non_selected = 0
        for i in range(0,self.number_clues):
            if(
                (i not in selectedIndices) # selects all notes that have not been considered as selected
              ):
                non_selected = non_selected+1
                avg_non_selected = avg_non_selected + self.state[i,0:self.embedding_size]
        if(non_selected != 0):
            avg_non_selected = avg_non_selected/non_selected


        # #simulate eye behaviour
        # # the division/multiplication  of the np.one array by the number determines how spread the value is
        # # i.e., if the provided array has large number, the distribution is somewhat even (all values are close to each other)
        # # else if the provided array has a smaller number, the distribution is skewed to have lesser numbers closer to 1
        # # this distribution generates values whose sum is = 1
        # dwellRatioArray = (np.random.dirichlet(np.ones(self.number_clues)/7.,size=1))[0]
        # self.state[:,self.embedding_size+2] = dwellRatioArray

        dwellRatioArray = np.zeros(self.nn_input_rows)
        dwellRatioArray[:self.number_clues] = (np.random.dirichlet(np.ones(self.number_clues)/7.,size=1))[0]
        self.state[:,self.embedding_size+2] = dwellRatioArray

        # # also for saccadeIn
        # saccadeInArray = (np.random.dirichlet(dwellRatioArray,size=1))[0]
        # self.state[:,self.embedding_size+3] = saccadeInArray

        saccadeInArray = np.zeros(self.nn_input_rows)
        saccadeInArray[:self.number_clues] = (np.random.dirichlet(np.ones(self.number_clues)/7.,size=1))[0]
        self.state[:,self.embedding_size+3] = saccadeInArray

        #simulate behaviour
        count_selected = 0

        # randomised state so that the first few selected post its when avg == 0
        # is not always the same
        random_iterator = list(range(0,self.number_clues))
        random.shuffle(random_iterator)
        for i in random_iterator:
            #if user has already selected maxPostPerStep postIts, break
            max_selected = random.randint(0,self.maxPostPerStep)
            if(count_selected >= max_selected):
                break

            ##########################################################
            # using Euclidean Distance
            # WORKS - returns different rewards for different episodes in test
            ##########################################################
            # dist_to_selected  = TextProcessor.euclideanDistance(avg_selected,self.state[i,0:self.embedding_size])
            # dist_to_non_seleceted = TextProcessor.euclideanDistance(avg_non_selected,self.state[i,0:self.embedding_size])
            #
            # # scale the distance based on post it state
            #
            # # scale selected distance based on current state (min = 0, max = 1, highlight = 2)
            # # higher state = less distance for selected
            # # if state = 1, divide by 1.1x i.e., larger distance
            # # state = 2, divide by 1.2x i.e., medium distance
            # # state = 3, divide by 1.3x i.e., smaller distance
            # scale_factor_selected = 1 + (self.state[i,self.embedding_size] + 1)/10
            # dist_to_selected = dist_to_selected/scale_factor_selected
            # # similarly scale for non_selected
            # # to keep it similar to the scaling above
            # # if state = 1, we need division by 1.3x i.e. smaller distance
            # # if state = 2, we need 1.2x
            # # if state = 3, we need 1.1x
            # # math.ceil (num_states/ postit state) -> works
            # scale_factor_non_selected = 1 + math.ceil(self.number_states/(self.state[i,self.embedding_size] + 1))/10
            # dist_to_non_seleceted = dist_to_non_seleceted/scale_factor_non_selected
            #
            # if(dist_to_selected < dist_to_non_seleceted):
            #     self.state[i,(self.embedding_size+1)] = 1
            #     count_selected+=1
            # else:
            #     self.state[i,(self.embedding_size+1)] = 0

            ##########################################################
            # using Cosine Similarity
            # doesn't actually work as the scores need to be normalized first
            # use angular similarity instead
            # DOES NOT WORK - returns same rewards for different episodes in test
            ##########################################################
            # similarity_selected = TextProcessor.cosine_similarity(avg_selected,self.state[i,0:self.embedding_size])
            # similarity_non_selected = TextProcessor.cosine_similarity(avg_non_selected,self.state[i,0:self.embedding_size])
            #
            # # adjust Similarity based on state
            #
            # # multiply selected by the state
            # # higher state = higher scaling
            # # if state = 1, 1.1x
            # # state = 2, 1.2x
            # # state = 3, 1.3x
            # scale_factor_selected = 1 + (self.state[i,self.embedding_size] + 1)/10
            # similarity_selected = similarity_selected * scale_factor_selected
            # # to keep it similar to the scaling above
            # # if state = 1, we need 1.3x
            # # if state = 2, we need 1.2x
            # # if state = 3, we need 1.1x
            # # math.ceil (num_states/ postit state) -> works
            # scale_factor_non_selected = 1 + math.ceil(self.number_states/(self.state[i,self.embedding_size] + 1))/10
            # similarity_non_selected = similarity_non_selected * scale_factor_non_selected
            #
            # if(similarity_selected > similarity_non_selected):
            #     self.state[i,(self.embedding_size+1)] = 1
            #     count_selected+=1
            # else:
            #     self.state[i,(self.embedding_size+1)] = 0

            ##########################################################
            # using angular distance
            # WORKS - returns different rewards for different episodes in test
            ##########################################################

            # angular_dist_to_selected  = TextProcessor.angular_distance(avg_selected,self.state[i,0:self.embedding_size])
            # angular_dist_to_non_seleceted = TextProcessor.angular_distance(avg_non_selected,self.state[i,0:self.embedding_size])
            #
            # # scale the distance based on post it state
            #
            # # scale selected distance based on current state (min = 0, max = 1, highlight = 2)
            # # higher state = less distance for selected
            # # if state = 1, divide by 1.1x i.e., larger distance
            # # state = 2, divide by 1.2x i.e., medium distance
            # # state = 3, divide by 1.3x i.e., smaller distance
            # scale_factor_selected = 1 + (self.state[i,self.embedding_size] + 1)/10
            # angular_dist_to_selected = angular_dist_to_selected/scale_factor_selected
            #
            # # similarly scale for non_selected
            # # to keep it similar to the scaling above
            # # if state = 1, we need division by 1.3x i.e. smaller distance
            # # if state = 2, we need 1.2x
            # # if state = 3, we need 1.1x
            # # math.ceil (num_states/ postit state) -> works
            # scale_factor_non_selected = 1 + math.ceil(self.number_states/(self.state[i,self.embedding_size] + 1))/10
            # angular_dist_to_non_seleceted = angular_dist_to_non_seleceted/scale_factor_non_selected
            #
            # if(angular_dist_to_selected < angular_dist_to_non_seleceted):
            #     self.state[i,(self.embedding_size+1)] = 1
            #     count_selected+=1
            # else:
            #     self.state[i,(self.embedding_size+1)] = 0

            ##########################################################
            # using angular Similarity
            ##########################################################

            angular_sim_to_selected  = TextProcessor.angular_similarity(avg_selected,self.state[i,0:self.embedding_size])
            angular_sim_to_non_selected = TextProcessor.angular_similarity(avg_non_selected,self.state[i,0:self.embedding_size])

            # scale the distance based on post it state

            # multiply selected by the state
            # higher state = higher scaling
            # if state = 1, 1.1x
            # state = 2, 1.2x
            # state = 3, 1.3x
            scale_factor_selected = 1 + (self.state[i,self.embedding_size] + 1)/10
            angular_sim_to_selected = angular_sim_to_selected * scale_factor_selected
            # to keep it similar to the scaling above
            # if state = 1, we need 1.3x
            # if state = 2, we need 1.2x
            # if state = 3, we need 1.1x
            # math.ceil (num_states/ postit state) -> works
            scale_factor_non_selected = 1 + math.ceil(self.number_states/(self.state[i,self.embedding_size] + 1))/10
            angular_sim_to_non_selected = angular_sim_to_non_selected * scale_factor_non_selected

            if(angular_sim_to_selected > angular_sim_to_non_selected):
                self.state[i,(self.embedding_size+1)] = 1
                count_selected+=1
            else:
                self.state[i,(self.embedding_size+1)] = 0

        #rewards
        reward = self.getReward_consider_changed_selected()
        # reward = self.getReward_consider_all()
        # reward = self.getReward_consider_changed()

        #check session length
        if(self.train_length <= 0):
            done = True
        else:
            done = False

        #info placeholder
        info = {}

        return self.state, reward, done, info


    def render(self):
        pass

    def reset(self):
        # self.state = self.start_state

        postIts = self.textProcessor.postItsFromFile_bert(self.filenames)
        self.state = BertTrainingEnv.getBox(postIts, self.embedding_size, self.nn_input_rows)

        self.train_length = 360
        return self.state

    ###################################################################
    # REWARD FUNCTIONS
    ###################################################################

    ## returns the reward
    ## considers all non selected post it notes
    ## considers only newly selected post it notes
    def getReward_consider_changed_selected(self):
        reward = 0
        for i in range(0, self.number_clues):
            # can have larger rewards/punishment as less items are selected than not
            if(self.state[i,(self.embedding_size+1)] == 1):
                #if the state has not changed, do not update reward.
                # only get reward for the first time an item is selected.
                if(abs(self.state[i,(self.embedding_size+1)] - self.previous_state[i,(self.embedding_size+1)]) == 0):
                    continue
                if(self.state[i,self.embedding_size] == 0):
                    reward =  reward - 1
                    # reward =  reward - 5
                if(self.state[i,self.embedding_size] == 1):
                    reward =  reward + 0.05
                    # reward =  reward - 1
                if(self.state[i,self.embedding_size] == 2):
                    reward =  reward + 2
                    # reward =  reward + 3
            # less rewards/punishment as more items are not selected per turn
            # contrary to previous case, get reward/punishment for every step items are not selected
            else:
                if(self.state[i,self.embedding_size] == 0):
                    reward =  reward + 0.1
                    # reward =  reward + 2
                if(self.state[i,self.embedding_size] == 1):
                    reward =  reward  + 0.05
                    # reward =  reward - 0.5
                if(self.state[i,self.embedding_size] == 2):
                    reward =  reward - 2
                    # reward =  reward - 10

        return reward

    ## returns the reward
    ## considers all post it notes
    def getReward_consider_all(self):
        reward = 0
        for i in range(0, self.number_clues):
            # can have larger rewards/punishment as less items are selected than not
            if(self.state[i,(self.embedding_size+1)] == 1):
                if(self.state[i,self.embedding_size] == 0):
                    # reward =  reward - 2
                    reward =  reward - (0.5 * self.number_clues)
                if(self.state[i,self.embedding_size] == 1):
                    # reward =  reward - 1
                    reward =  reward + (0.01 * self.number_clues)
                if(self.state[i,self.embedding_size] == 2):
                    # reward =  reward + 2
                    reward =  reward + (1 * self.number_clues)
            # less rewards/punishment as more items are not selected per turn
            else:
                if(self.state[i,self.embedding_size] == 0):
                    # reward =  reward + 1
                    reward =  reward + 0
                if(self.state[i,self.embedding_size] == 1):
                    # reward =  reward + 0
                    reward =  reward - 0.01
                if(self.state[i,self.embedding_size] == 2):
                    # reward =  reward - 1
                    reward =  reward - 1

        return reward

    ## returns the reward
    ## considers only post it notes whose isSelected(number_topics + 1) value has changed
    ## this just does not work -  no learning happens
    def getReward_consider_changed(self):
        reward = 0

        state_shape = list(self.state.shape) # change to list to unmute
        state_shape[-1] = state_shape[-1]+1 #add column to indicate if changed from last state
        state_shape = tuple(state_shape) # change back to tuple
        state_with_changed = np.ndarray(state_shape)
        state_with_changed[:,:-1] = self.state

        # set last index to 1 if there was a change of the isSelected property
        state_with_changed[:,-1] = abs(self.state[:,(self.embedding_size + 1)] - self.previous_state[:,(self.embedding_size + 1)])

        only_changed = np.array([x for x in state_with_changed if x[-1] == 1])

        for i in range(0, len(only_changed)):
            if(only_changed[i,(self.embedding_size+1)] == 1):
                if(only_changed[i,self.embedding_size] == 0):
                    # reward =  reward - 2
                    reward - (0.5 * self.number_clues)
                if(only_changed[i,self.embedding_size] == 1):
                    # reward =  reward - 1
                    reward =  reward + (0.01 * self.number_clues)
                if(only_changed[i,self.embedding_size] == 2):
                    # reward =  reward + 2
                    reward =  reward + (1 * self.number_clues)
            else:
                if(only_changed[i,self.embedding_size] == 0):
                    # reward =  reward + 1
                    reward =  reward - 0.01
                if(only_changed[i,self.embedding_size] == 1):
                    # reward =  reward + 0
                    reward =  reward - 0.02
                if(only_changed[i,self.embedding_size] == 2):
                    # reward =  reward - 1
                    reward =  reward - 0.05

        return reward

    ###################################################################
    # CLASS METHODS
    ###################################################################
    #supporting function to get the observation space as box
    @classmethod
    def getBox(cls, postItDict, embedding_size, nn_input_rows):
        # postItMatrix = np.zeros(shape = (len(postItDict), embedding_size + 4), dtype = np.float32)
        postItMatrix = np.zeros(shape = (nn_input_rows, embedding_size + 4), dtype = np.float32)
        for postItId in postItDict:
            postItMatrix[postItId,0:embedding_size] = postItDict[postItId].topics
            postItMatrix[postItId,embedding_size] = postItDict[postItId].state
            postItMatrix[postItId,(embedding_size+1)] = postItDict[postItId].isSelected
            postItMatrix[postItId,(embedding_size+2)] = postItDict[postItId].lastEyeDwellRatio
            postItMatrix[postItId,(embedding_size+3)] = postItDict[postItId].saccadeInRatio

        return postItMatrix
