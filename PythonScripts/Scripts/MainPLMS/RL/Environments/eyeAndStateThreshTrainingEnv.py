from collections import OrderedDict
from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiDiscrete, Tuple
import numpy as np, numpy.random
import random
import copy
import sys

sys.path.append("../..")

from TextProcessor.textProcessor import TextProcessor

##################################################################
#generates topic probabilities for each episode
##################################################################
class EyeAndStateThreshTrainingEnv(Env):
    # def _init__(self, number_clues, number_topics, number_states, start_state):
    def __init__(self, number_clues, number_topics, number_states, filenames, textProcessor, dwellEpsilon, maxPostPerStep): # we pass the files as topic probs change. Only useful for training

        self.number_clues = number_clues
        self.number_topics = number_topics
        self.number_states = number_states
        self.dwellEpsilon = dwellEpsilon
        self.maxPostPerStep = maxPostPerStep

        self.filenames = filenames
        self.textProcessor = textProcessor
        postIts, ldaModel, ldaDictionary = self.textProcessor.postItsFromFile(self.filenames, self.number_topics)
        self.start_state = EyeAndStateThreshTrainingEnv.getBox(postIts,self.number_topics)

        ##################################################################
        #BOX
        ##################################################################

        #number of clues x (number of topics) + 1(states) + 1 (isSelected) + 1 (postIt eye dwell time ratio)
        obs_high = np.ones(shape = (self.number_clues,self.number_topics+3), dtype = np.float32)
        obs_high[:,self.number_topics] = np.float32(2)
        obs_low = np.zeros(shape = (self.number_clues,self.number_topics+3), dtype = np.float32)
        self.observation_space = Box(low=obs_low, high = obs_high, dtype = np.float32)

        ##################################################################

        # 0 -> low_thresh, 1 -> high_thresh and 2 -> dwell_thre
        self.action_space = Box(low = np.float32(0), high = np.float32(1), shape = (3,), dtype = np.float32)

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
    # state[:,number_topics] -> states
    # state[:,number_topics+1] -> isSelected
    # state[:,number_topics+2] -> dwellTimeRatio
    def step(self, action):
        #set previous state
        self.previous_state = copy.deepcopy(self.state)

        # this is dodgy, the sigmoid output layer sometimes gives values slightly larger than 1 or slightly smaller 0
        action = np.clip(action,0,1)

        assert self.action_space.contains(action)

        # create list of actions
        actions = np.ones(self.number_clues)
        actions = actions*(-1)

        #send action to HoloLens

        #simulate user behaviour
        #blocking wait for new state from HoloLens

        # calculate average topic probability of
        # Selected topics
        # and topics that have been looked at for a long enough time
        avg = np.zeros(shape = (self.number_topics))
        selected = 0
        for i in range(0,self.number_clues):
            if(
                    (self.state[i,(self.number_topics+1)] == 1)
                    and (self.state[i,(self.number_topics+2)] >= action[2])
                ):
                selected = selected+1
                avg = avg + self.state[i,0:self.number_topics]
        if(selected != 0):
            avg = avg/selected

        for i in range(0,self.number_clues):
            # get euclidean distance between note and avg
            # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
            # dist = np.linalg.norm(avg - self.state[i,0:self.number_topics])
            dist = TextProcessor.euclideanDistance(avg,self.state[i,0:self.number_topics])
            if(dist <= action[0]):
                actions[i] = 0
            if(action[0]<dist<action[1]):
                actions[i] = 1
            if(dist >= action[1]):
                actions[i] = 2

        self.state[:,self.number_topics] = actions
        self.train_length -= 1

        # set the maximum value to 1 and the rest to 0
        # the index to which 1 belongs to indicates which topic has the highest probability
        avg_binary = [1 if x == max(avg) else 0 for x in avg]

        #simulate eye behaviour
        # the division/multiplication  of the np.one array by the number determines how spread the value is
        # i.e., if the provided array has large number, the distribution is somewhat even (all values are close to each other)
        # else if the provided array has a smaller number, the distribution is skewed to have lesser numbers closer to 1
        # this distribution generates values whose sum is = 1
        dwellRatioArray = (np.random.dirichlet(np.ones(self.number_clues)/7.,size=1))[0]
        self.state[:,self.number_topics+2] = dwellRatioArray


        #simulate behaviour
        count_selected = 0

        # randomised state so that the first few selected post its when avg == 0
        # is not always the same
        random_iterator = list(range(0,self.number_clues))
        random.shuffle(random_iterator)
        for i in random_iterator:
            #if user has already selected maxPostPerStep postIts, break
            max_selected = random.randint(1,self.maxPostPerStep)
            if(count_selected >= max_selected):
                break

            prob =  (self.state[i,0:self.number_topics] * (self.state[i,self.number_topics]+1))/self.number_states

            #set to binary
            # the index to which 1 belongs to indicates which topic has the highest probability
            prob_binary = [1 if x == max(prob) else 0 for x in prob]


            # only consider notes that are not selected.
            # i.e., users do not remove notes onces Selected
            #s seems to break learning
            # if(self.state[i,(self.number_topics+1)] != 1):

            # the first condition is only useful when no post its have been selected at all
            # this is because
            # avg_binary = [1 if x == max(avg) else 0 for x in avg]
            # will always return 1 for all elements as the initial avg will contain the same elements i.e., 0
            if(
                    (np.all(avg_binary == np.ones(self.number_topics)) and (self.state[i,self.number_topics+2] > self.dwellEpsilon))
                    or (np.all(prob_binary == avg_binary) and (self.state[i,self.number_topics+2] > self.dwellEpsilon))
                ):
                self.state[i,(self.number_topics+1)] = 1
                count_selected+=1
            # the following is only needed if
            # we consider both selected and not selected notes and
            # we consider cases where user removes notes
            # this is unlikely
            else:
                self.state[i,(self.number_topics+1)] = 0

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

        postIts, ldaModel, ldaDictionary = self.textProcessor.postItsFromFile(self.filenames, self.number_topics)
        self.state = EyeAndStateThreshTrainingEnv.getBox(postIts,self.number_topics)

        self.train_length = 360
        return self.state

    ###################################################################
    # REWARD FUNCTIONS
    ###################################################################

    ## returns the reward
    ## considers all post it notes
    def getReward_consider_changed_selected(self):
        reward = 0
        for i in range(0, self.number_clues):
            # can have larger rewards/punishment as less items are selected than not
            if(self.state[i,(self.number_topics+1)] == 1):
                #if the state has not changed, do not update reward.
                # only get reward for the first time an item is selected.
                if(abs(self.state[i,(self.number_topics+1)] - self.previous_state[i,(self.number_topics+1)]) == 0):
                    continue
                if(self.state[i,self.number_topics] == 0):
                    # reward =  reward - 2
                    reward =  reward - (0.5 * self.number_clues)
                if(self.state[i,self.number_topics] == 1):
                    # reward =  reward - 1
                    reward =  reward + (0.01 * self.number_clues)
                if(self.state[i,self.number_topics] == 2):
                    # reward =  reward + 2
                    reward =  reward + (0.2 * self.number_clues)
            # less rewards/punishment as more items are not selected per turn
            # contrary to previous case, get reward/punishment for every step items are not selected
            else:
                if(self.state[i,self.number_topics] == 0):
                    # reward =  reward + 1
                    reward =  reward - 0.01
                if(self.state[i,self.number_topics] == 1):
                    # reward =  reward + 0
                    reward =  reward - 0.5
                if(self.state[i,self.number_topics] == 2):
                    # reward =  reward - 1
                    reward =  reward - 1

        return reward

    ## returns the reward
    ## considers all post it notes
    def getReward_consider_all(self):
        reward = 0
        for i in range(0, self.number_clues):
            # can have larger rewards/punishment as less items are selected than not
            if(self.state[i,(self.number_topics+1)] == 1):
                if(self.state[i,self.number_topics] == 0):
                    # reward =  reward - 2
                    reward =  reward - (0.5 * self.number_clues)
                if(self.state[i,self.number_topics] == 1):
                    # reward =  reward - 1
                    reward =  reward + (0.01 * self.number_clues)
                if(self.state[i,self.number_topics] == 2):
                    # reward =  reward + 2
                    reward =  reward + (1 * self.number_clues)
            # less rewards/punishment as more items are not selected per turn
            else:
                if(self.state[i,self.number_topics] == 0):
                    # reward =  reward + 1
                    reward =  reward + 0
                if(self.state[i,self.number_topics] == 1):
                    # reward =  reward + 0
                    reward =  reward - 0.01
                if(self.state[i,self.number_topics] == 2):
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
        state_with_changed[:,-1] = abs(self.state[:,(self.number_topics + 1)] - self.previous_state[:,(self.number_topics + 1)])

        only_changed = np.array([x for x in state_with_changed if x[-1] == 1])

        for i in range(0, len(only_changed)):
            if(only_changed[i,(self.number_topics+1)] == 1):
                if(only_changed[i,self.number_topics] == 0):
                    # reward =  reward - 2
                    reward - (0.5 * self.number_clues)
                if(only_changed[i,self.number_topics] == 1):
                    # reward =  reward - 1
                    reward =  reward + (0.01 * self.number_clues)
                if(only_changed[i,self.number_topics] == 2):
                    # reward =  reward + 2
                    reward =  reward + (1 * self.number_clues)
            else:
                if(only_changed[i,self.number_topics] == 0):
                    # reward =  reward + 1
                    reward =  reward - 0.01
                if(only_changed[i,self.number_topics] == 1):
                    # reward =  reward + 0
                    reward =  reward - 0.02
                if(only_changed[i,self.number_topics] == 2):
                    # reward =  reward - 1
                    reward =  reward - 0.05

        return reward

    ###################################################################
    # CLASS METHODS
    ###################################################################
    #supporting function to get the observation space as box
    @classmethod
    def getBox(cls, postItDict, number_topics):
        postItMatrix = np.zeros(shape = (len(postItDict), number_topics + 3), dtype = np.float32)
        for postItId in postItDict:
            postItMatrix[postItId,0:number_topics] = [y for (x,y) in postItDict[postItId].topics]
            postItMatrix[postItId,number_topics] = postItDict[postItId].state
            postItMatrix[postItId,(number_topics+1)] = postItDict[postItId].isSelected
            postItMatrix[postItId,(number_topics+2)] = postItDict[postItId].lastEyeDwellRatio

        return postItMatrix


    #supporting function to get the observation space as dict
    @classmethod
    def getDict(cls, postItDict):
        topics_array = []
        states_array = []
        isSelected_array = []

        for postItId in postItDict:
            if(len(topics_array) == 0):
                topics_array = [y for (x,y) in postItDict[postItId].topics]
                states_array = [postItDict[postItId].state]
                isSelected_array = [postItDict[postItId].isSelected]
            else:
                topics_array = np.vstack((
                        topics_array,
                        [y for (x,y) in postItDict[postItId].topics]
                    ))
                states_array.append(postItDict[postItId].state)
                isSelected_array.append(postItDict[postItId].isSelected)

        obsevartion_dict = OrderedDict({
            "topic_probs": topics_array,
            "states": states_array,
            "isSelected": isSelected_array
        })

        return obsevartion_dict
