#  TODO  #
# reward calculation and simulation needs to be updated
# see RL/eyeAndStateTrainingEnv

from collections import OrderedDict
from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiDiscrete, Tuple
import numpy as np
import random

##################################################################
#generates topic probabilities for each episode
##################################################################
class SensemakingTrainingEnv(Env):
    # def _init__(self, number_clues, number_topics, number_states, start_state):
    def __init__(self, number_clues, number_topics, number_states, filenames, textProcessor, epsilon, maxPostPerStep): # we pass the files as topic probs change. Only useful for training

        self.number_clues = number_clues
        self.number_topics = number_topics
        self.number_states = number_states
        self.epsilon = epsilon
        self.maxPostPerStep = maxPostPerStep

        self.filenames = filenames
        self.textProcessor = textProcessor
        postIts, ldaModel, ldaDictionary = self.textProcessor.postItsFromFile(self.filenames, self.number_topics)
        self.start_state = SensemakingTrainingEnv.getBox(postIts,self.number_topics)

        ##################################################################
        # DICTIONARY
        ##################################################################

        # May not be compatible input for an ANN
        # may need to flatten -> gym.spaces.flatten(obs_space,obs_space.sample())

        # self.observation_space = Dict({
        #     "topic_probs": Box(
        #                             low = np.float32(0),
        #                             high = np.float32(1),
        #                             shape = (self.number_clues, self.number_topics),
        #                             dtype = np.float32
        #                         ),
        #     "states": Box(
        #                     low = np.int32(0),
        #                     high = np.int32(2),
        #                     shape = (self.number_clues,),
        #                     dtype = np.int32
        #                 ),
        #     "isSelected": Box(
        #                         low = np.int32(0),
        #                         high = np.int32(1),
        #                         shape = (self.number_clues,),
        #                         dtype = np.int32
        #                     )
        # })

        ##################################################################
        #BOX
        ##################################################################

        #number of clues x (number of topics) + 1(states) + 1 (isSelected)
        obs_high = np.ones(shape = (self.number_clues,self.number_topics+2), dtype = np.float32)
        obs_high[:,self.number_topics] = np.float32(2)
        obs_low = np.zeros(shape = (self.number_clues,self.number_topics+2), dtype = np.float32)
        self.observation_space = Box(low=obs_low, high = obs_high, dtype = np.float32)

        ##################################################################

        ##to get the number of states, we need to flatten and then find shape[0]
        ## gym.spaces.flatten(obs_space,obs_space.smaple()).shape[0]
        ## or gym.spaces.flatdim(obs_space)

        ##################################################################

        ##################################################################
        #for eye tracking
        # may not be required as eye tracking dwell time will be put as an extra parameter
        ##################################################################

        #represents the proportion of time spent on postIt with id = index
        # proportion = time spent/ sampling time
        # self.observation_space = Box(
        #     low = np.float32(0),
        #     high = np.float32(1),
        #     shape = (number_clues,1),
        #     dtype = np.float32
        # )

        ##################################################################

        self.action_space = Box(low = np.int32(0), high = np.int32(2), shape = (self.number_clues,), dtype = np.int32)


        ##################################################################

        #for eye tracking
        # add or substract action from eye tracking selection threshold
        # self.action_space = Box(low = np.float32(-1), high = np.float(1), shape = (1,1), dtype = np.float32)

        ##################################################################

        ##################################################################
        # test
        # self.action_space = MultiDiscret([number_clues,3])
        ##################################################################

        #setting state to start_state
        self.state = self.start_state

        #length
        self.train_length = 60

    ##################################################################
    # WITH DICT SPACE
    # used when the observation space is represented by gym.spaces.Dict (see above)
    ##################################################################
    # def step(self, action):
    #     self.state["states"] = action
    #     self.train_length -= 1
    #
    #     #send action to HoloLens
    #
    #     #simulate user behaviour
    #     #blocking wait for new state from HoloLens
    #
    #     #calculate average topic probability of Selected topics
    #     avg = np.zeros(shape = (self.number_topics))
    #     selected = 0
    #     for i in range(0,self.number_clues):
    #         if(self.state["isSelected"][i] == 1):
    #             selected = selected+1
    #             avg = avg + self.state["topic_probs"][i]
    #     avg = avg/selected
    #
    #     #simulate behaviour
    #     epsilon = 0.8
    #     for i in range(0,self.number_clues):
    #         for j in range(0,self.number_topics):
    #             #calculates the probability of a state to get selected based on its topic and post it state
    #             prob = (self.state["topic_probs"][i,j] * self.state["states"][i])/self.number_states
    #             if((prob > avg[j]) or (random.random() > epsilon)):
    #                 self.state["isSelected"][i] = 1
    #             else:
    #                 self.state["isSelected"][i] = 0
    #
    #     #rewards
    #     reward = 0
    #     for i in range(0, self.number_clues):
    #         if(self.state["isSelected"][i] == 1):
    #             if(self.state["states"][i] == 0):
    #                 reward =  reward - 2
    #             if(self.state["states"][i] == 1):
    #                 reward =  reward - 1
    #             if(self.state["states"][i] == 2):
    #                 reward =  reward + 2
    #         else:
    #             if(self.state["states"][i] == 0):
    #                 reward =  reward + 1
    #             if(self.state["states"][i] == 1):
    #                 reward =  reward + 0
    #             if(self.state["states"][i] == 2):
    #                 reward =  reward - 1
    #
    #     #check session length
    #     if(self.train_length <= 0):
    #         done = True
    #     else:
    #         done = False
    #
    #     #info placeholder
    #     info = {}
    #
    #     return self.state, reward, done, info

    ##################################################################
    # WITH BOX SPACE
    # used when observation_space is represented by gy.spaces.Box
    ##################################################################
    # state[:,0:number_topics] -> topic_probs
    # state[:,number_topics] -> states
    # state[:,number_topics+1] -> isSelected
    def step(self, action):
        self.state[:,self.number_topics] = action
        self.train_length -= 1

        #send action to HoloLens

        #simulate user behaviour
        #blocking wait for new state from HoloLens

        #calculate average topic probability of Selected topics
        avg = np.zeros(shape = (self.number_topics))
        selected = 0
        for i in range(0,self.number_clues):
            if(self.state[i,(self.number_topics+1)] == 1):
                selected = selected+1
                avg = avg + self.state[i,0:self.number_topics]
        if(selected != 0):
            avg = avg/selected

        # set the maximum value to 1 and the rest to 0
        # the index to which 1 belongs to indicates which topic has the highest probability
        avg_binary = [1 if x == max(avg) else 0 for x in avg]

        #simulate behaviour
        count_selected = 0
        for i in range(0,self.number_clues):
            #if user has already selected maxPostPerStep postIts, break
            max_selected = random.randint(1,self.maxPostPerStep)
            if(count_selected >= max_selected):
                break

            prob =  (self.state[i,0:self.number_topics] * self.state[i,self.number_topics]+1)/self.number_states

            #set to binary
            # the index to which 1 belongs to indicates which topic has the highest probability
            prob_binary = [1 if x == max(prob) else 0 for x in prob]

            if(
                    (prob_binary == avg_binary)
                    or (random.random() > self.epsilon)
                ):
                self.state[i,(self.number_topics+1)] = 1
                count_selected+=1
            else:
                self.state[i,(self.number_topics+1)] = 0

        ####################################################################################################################################
        # This version is actually wrong as it will select a post it note if any topic prob > avg_topic prob
        # it also will deselect a topic immediately if the current topic considered is less than the avg_topic prob

        #simulate behaviour
        # count_selected = 0
        # for i in range(0,self.number_clues):
        #     #if user has already selected maxPostPerStep postIts, break
        #     max_selected = random.randint(1,self.maxPostPerStep)
        #     if(count_selected >= max_selected):
        #         break
        #
        #     for j in range(0,self.number_topics):
        #         #calculates the probability of a state to get selected based on its topic and post it state
        #         # * post it state (0,1,2) + 1
        #         # where stat[i,j] gives us the topic probabilities
        #         # state[i,self.num_topics] gives us the postItState (0,1,2) for postIt i
        #         prob = (self.state[i,j] * (self.state[i,self.number_topics]+1))/self.number_states
        #         if((prob > avg[j]) or (random.random() > self.epsilon)):
        #             self.state[i,(self.number_topics+1)] = 1
        #             count_selected+=1
        #         else:
        #             self.state[i,(self.number_topics+1)] = 0

        ####################################################################################################################################

        #rewards
        reward = 0
        for i in range(0, self.number_clues):
            if(self.state[i,(self.number_topics+1)] == 1):
                if(self.state[i,self.number_topics] == 0):
                    reward =  reward - 2
                if(self.state[i,self.number_topics] == 1):
                    reward =  reward - 1
                if(self.state[i,self.number_topics] == 2):
                    reward =  reward + 2
            else:
                if(self.state[i,self.number_topics] == 0):
                    reward =  reward + 1
                if(self.state[i,self.number_topics] == 1):
                    reward =  reward + 0
                if(self.state[i,self.number_topics] == 2):
                    reward =  reward - 1

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
        self.state = SensemakingTrainingEnv.getBox(postIts,self.number_topics)

        self.train_length = 60
        return self.state

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

    #supporting function to get the observation space as box
    @classmethod
    def getBox(cls, postItDict, number_topics):
        postItMatrix = np.zeros(shape = (len(postItDict), number_topics + 2), dtype = np.float32)
        for postItId in postItDict:
            postItMatrix[postItId,0:number_topics] = [y for (x,y) in postItDict[postItId].topics]
            postItMatrix[postItId,number_topics] = postItDict[postItId].state
            postItMatrix[postItId,(number_topics+1)] = postItDict[postItId].isSelected

        return postItMatrix
