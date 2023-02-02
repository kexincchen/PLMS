from collections import OrderedDict
from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiDiscrete, Tuple
import numpy as np, numpy.random
import random
import sys
import copy

sys.path.append("..")

from Network.udp import UDPSocket
from Entity.jsonMessage import JsonMessage

from Entity.actionMap import ActionMap
from Entity.actionMapList import ActionMapList
##################################################################
#generates topic probabilities for each episode
##################################################################
class BertEnv(Env):
    def __init__(self, number_clues, nn_input_rows, embedding_size, number_states, start_state, udpSocket, udp_ip_remote, udp_port_remote):

        self.number_clues = number_clues
        self.nn_input_rows = nn_input_rows
        self.embedding_size = embedding_size
        self.number_states = number_states

        self.udpSocket = udpSocket
        self.udp_ip_remote = udp_ip_remote
        self.udp_port_remote = udp_port_remote

        self.start_state = start_state

        ##################################################################
        #BOX
        ##################################################################

        #number of clues x (embedding size) + 1(states) + 1 (isSelected) + 1 (postIt eye dwell time ratio) + 1 (saccadeIn Ratio)
        # obs_high = np.ones(shape = (self.number_clues,self.embedding_size+4), dtype = np.float32)
        # obs_high[:,self.embedding_size] = np.float32(2)
        # obs_low = np.zeros(shape = (self.number_clues,self.embedding_size+4), dtype = np.float32)
        # self.observation_space = Box(low=obs_low, high = obs_high, dtype = np.float32)
        #
        # obs_high = np.ones(shape = (self.nn_input_rows,self.embedding_size+4), dtype = np.float32)
        # obs_high[:,self.embedding_size] = np.float32(2)
        # obs_low = np.zeros(shape = (self.nn_input_rows,self.embedding_size+4), dtype = np.float32)
        # self.observation_space = Box(low=obs_low, high = obs_high, dtype = np.float32)

        ##################################################################
        # For non softmax version
        # self.action_space = Box(low = np.int32(0), high = np.int32(2), shape = (self.number_clues,), dtype = np.int32)

        # self.action_space = Box(low = np.int32(0), high = np.int32(2), shape = (self.nn_input_rows,), dtype = np.int32)

        # for SOFTMAX
        # self.action_space = Box(low = np.int32(0), high = np.int32(1), shape = (self.number_clues,self.number_states), dtype = np.float32)

        self.action_space = Box(low = np.int32(0), high = np.int32(1), shape = (self.nn_input_rows,self.number_states), dtype = np.float32)
        ##################################################################

        #setting state to start_state
        self.state = self.start_state
        self.previous_state = copy.deepcopy(self.state)

        # has the first state been observed
        self.initialState = False

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
        if(not self.initialState):
            #blocking wait for new state from HoloLens
            while(len(self.udpSocket.obsQueue) == 0):
                pass

            # set the state to the first state
            self.state = self.udpSocket.obsQueue.pop(0)
            self.initialState = True

        # set previous state to the state before we get an update
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


        #send action to HoloLens
        # aMapList = BertEnv.convertActionsToActionMapList(action)
        aMapList = BertEnv.convertActionsToActionMapList(action[:self.number_clues])

        ########################
        # construct JSON message and send to HoloLens

        # this throws an error of message too long.
        # jm = JsonMessage("ActionMapList",aMapList)
        # self.udpSocket.sendUDPMsg(jm, self.udp_ip_remote, self.udp_port_remote)

        for aMap in aMapList.values:
            jm = JsonMessage("ActionMap", aMap)
            self.udpSocket.sendUDPMsg(jm, self.udp_ip_remote, self.udp_port_remote)


        #blocking wait for new state from HoloLens
        while(len(self.udpSocket.obsQueue) == 0):
            pass

        self.state = self.udpSocket.obsQueue.pop(0)

        #rewards
        reward = self.getReward_consider_changed_selected()

        #check session length
        if(self.udpSocket.isEpisodeDone == True):
            print("Environment Episode Done")
            done = True
        else:
            done = False

        #info placeholder
        info = {}

        return self.state, reward, done, info


    def render(self):
        pass

    def reset(self):
        self.state = self.start_state
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

    #supporting class to convert our action_space to a actionMapList to send to the HoloLens
    @classmethod
    def convertActionsToActionMapList(cls,actions):
        aMapList = ActionMapList()
        for i in range(len(actions)):
            aMap = ActionMap(id = i, action = int(actions[i]))
            aMapList.values.append(aMap)
        return aMapList
