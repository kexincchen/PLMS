import sys,getopt
import time
import os.path
import numpy as np
import gym
import math
import matplotlib.pyplot as plt

sys.path.append("..")

from tensorflow.keras.optimizers import Adam

import tensorflow as tf
print(tf.__version__)


# from Network.udp import UDPSocket
# from Entity.jsonMessage import JsonMessage
# from Entity.postItContent import PostItContent
# from Entity.postItNumber import PostItNumber
# from Entity.postItData import PostItData
# from Entity.postItVal import PostItVal
# from Entity.postItValList import PostItValList
# from Entity.actionMap import ActionMap
# from Entity.actionMapList import ActionMapList
# from Entity.completionTime import CompletionTime

from TextProcessor.textProcessor import TextProcessor

# from RL.Environments.sensemakingTrainingEnv import SensemakingTrainingEnv
# from RL.Environments.sensemakingTrainingEnv2 import SensemakingTrainingEnv2
# from RL.Environments.eyeAndStateTrainingEnv import EyeAndStateTrainingEnv
# from RL.Environments.eyeAndStateEnv import EyeAndStateEnv
# from RL.Environments.eyeAndStateThreshTrainingEnv import EyeAndStateThreshTrainingEnv
# from RL.Environments.eyeAndStateThreshEnv import EyeAndStateThreshEnv
from RL.Environments.bertTrainingEnv import BertTrainingEnv
# from RL.Environments.bertEnv import BertEnv

from RL.Solvers.randomSolver import RandomSolver
# from RL.Solvers.deepQLearning import DeepQLearning
from RL.Solvers.ddpgLearning import DDPGLearning
# from RL.Solvers.clustering import Clustering

####################################################################################################################################
# SETTINGS

# to keep usign he trained params for any number of clues
# we need to keep the NN input row and columns the same
# the columns are determined by the sentence_embeddings + number of user parameters and task parmeters (gaze + task state)
# for rows we use a statis and pad with zeros
NN_INPUT_ROW = 60 # estimating that 60 will be the max post it notes in any murder mystery

NUM_STATES = 3

# for LDA
NUM_TOPICS = 3

# for BERT
EMBEDDING_SIZE = 768

##################################################################
# FOR TRAINING ENVIRONMENTS
##################################################################
#the dummy users EXPLOITATION probability of the topic probabilities
#as opposed to EXPLORATION -> 1-EPSILON
# only for training with non-eye tracking data
EPSILON = 0.7

# dwell time ratio (based on window size) on post it should be larger than this value
MIN_DWELL_EPSILON = 0.1

# maximum number of post-its that can be selected per step by the dummy user
MAX_POST_PER_STEP = 3

#method to use: AR or RL or CLUSTERING or RL_CLUSTER
method = "RL"

#stanford murder mystery
filename1 = "../../../Data/murder_mystery/stanford_mm.txt"
#stanford murder mystery DISTRACTIONS
filename2 = "../../../Data/murder_mystery/stanford_mm_distractions.txt"

#bank robbery mystery
# filename1 = "../../../Data/murder_mystery/bank_robbery.txt"

FILENAMES = []
FILENAMES.append(filename1)
FILENAMES.append(filename2)

# for storing the participant ID
pid = -1

####################################################################################################################################
def SetUpTrainingEnv(postIts,textProcessor):
    ######################################################
    # LDA
    ######################################################
    ##training env with simulated eye tracking Data
    # env = EyeAndStateTrainingEnv(
    #                             len(postIts),
    #                             NUM_TOPICS,
    #                             NUM_STATES,
    #                             FILENAMES,
    #                             textProcessor,
    #                             MIN_DWELL_EPSILON,
    #                             MAX_POST_PER_STEP
    #                         )

    ##training env with simulated eye tracking Data
    # outputs 3 threshold
    # 0 -> any distance less than this will result in minimize action
    # 1 -> any distance greater than this will result in highlight action
    # 2 -> Threshold to determine if gazed post it will be considered
    # env = EyeAndStateThreshTrainingEnv(
    #                             len(postIts),
    #                             NUM_TOPICS,
    #                             NUM_STATES,
    #                             FILENAMES,
    #                             textProcessor,
    #                             MIN_DWELL_EPSILON,
    #                             MAX_POST_PER_STEP
    #                         )
    #

    ######################################################
    # BERT SENTENCE EMBEDDINGS
    ######################################################
    env = BertTrainingEnv(
                                len(postIts),
                                NN_INPUT_ROW,
                                EMBEDDING_SIZE,
                                NUM_STATES,
                                FILENAMES,
                                textProcessor,
                                MIN_DWELL_EPSILON,
                                MAX_POST_PER_STEP
                            )

    return env
####################################################################################################################################

####################################################################################################################################

#using a random solver -> picks random actions
# RandomSolver.solve(env)

def buildDDPGModel(env):
    # Dense First Network
    # actor, critic, critic_action_input = DDPGLearning.build_model(
    #                                                                 env.observation_space,
    #                                                                 env.action_space
    #                                                             )
    # actor.summary()
    # critic.summary()
    # return actor, critic, critic_action_input

    # SOFTMAX
    # to use this variant - changes need to be made in:
    # bertTrainingEnv - RL/Environment
        # change action_space in __init__()
        # change postprocessing of action in step()
    # bertEnv - RL/Environment
        # change action_space in __init__()
        # change postprocessing of action in step()
    actor, critic, critic_action_input = DDPGLearning.build_model_softmax(
                                                                    env.observation_space,
                                                                    env.action_space
                                                                )
    actor.summary()
    critic.summary()
    return actor, critic, critic_action_input

    # Dense First Network with Discretization
    # actor, critic, critic_action_input = DDPGLearning.build_model_discrete(
    #                                                                 env.observation_space,
    #                                                                 env.action_space
    #                                                             )
    # actor.summary()
    # critic.summary()
    # return actor, critic, critic_action_input

    # Dense Only Network (FlatEnd)
    # actor, critic, critic_action_input = DDPGLearning.build_model_flatEnd(
    #                                                                 env.observation_space,
    #                                                                 env.action_space
    #                                                             )
    # actor.summary()
    # critic.summary()
    # return actor, critic, critic_action_input

    # Flat first - does not work well
    # actor, critic, critic_action_input = DDPGLearning.build_model_flatFirst(
    #                                                                 env.observation_space,
    #                                                                 env.action_space
    #                                                             )
    # actor.summary()
    # critic.summary()
    # return actor, critic, critic_action_input

    # Threshold based output - does not work well
    # actor, critic, critic_action_input = DDPGLearning.build_model_threshold(
    #                                                                 env.observation_space,
    #                                                                 env.action_space
    #                                                             )
    # actor.summary()
    # critic.summary()
    # return actor, critic, critic_action_input

def buildDDPGAgent(actor, critic, critic_action_input, env):
    agent = DDPGLearning.build_agent(actor, critic, critic_action_input, env.action_space)
    agent.compile(Adam(lr = 1e-3), metrics = ['mae'])

    return agent

def trainDDPGAgent(agent, env, weight_file):
    prefix, suffix = weight_file.split(".")

    actor_weight_index = prefix + "_actor"+ "."+ suffix + '.index'
    actor_weight_data = prefix + "_actor"+ "."+ suffix + '.data-00000-of-00001'

    critic_weight_index = prefix + "_critic"+ "."+ suffix + '.index'
    critic_weight_data = prefix + "_critic"+ "."+ suffix + '.data-00000-of-00001'

    if(
            (os.path.exists(actor_weight_index) and (os.path.exists(actor_weight_data)))
            and (os.path.exists(critic_weight_index) and (os.path.exists(critic_weight_data)))
        ):
        # print("file exists")
        agent.load_weights(weight_file)

        # to load history
        history = np.load('my_history_eval.npy', allow_pickle='TRUE').item()
    else:
        # print("file does not exists")
        # agent.fit(env, nb_episodes = 138, visualize = False, verbose = 1)

        # for TESTING
        history = agent.fit(env, nb_episodes = 138, visualize = False, verbose = 1) # possibly do this for 1 then 2 then 3 ,.... episodes and plot rewards
        print(history.history.keys())
        print(history.history)
        history = history.history
        # to save history
        np.save('my_history_eval.npy', history)

        agent.save_weights(weight_file, overwrite = True)

    plt.plot(history['episode_reward'])
    plt.title('Rewards')
    plt.ylabel('episode_reward')
    plt.xlabel('episodes')
    plt.show()

    return agent

def loadWeights(agent, weight_file):
    prefix, suffix = weight_file.split(".")

    actor_weight_index = prefix + "_actor"+ "."+ suffix + '.index'
    actor_weight_data = prefix + "_actor"+ "."+ suffix + '.data-00000-of-00001'

    critic_weight_index = prefix + "_critic"+ "."+ suffix + '.index'
    critic_weight_data = prefix + "_critic"+ "."+ suffix + '.data-00000-of-00001'

    if(
            (os.path.exists(actor_weight_index) and (os.path.exists(actor_weight_data)))
            and (os.path.exists(critic_weight_index) and (os.path.exists(critic_weight_data)))
        ):
        # print("file exists")
        agent.load_weights(weight_file)

####################################################################################################################################
#test agent
def testAgent(agent,env):
    scores = agent.test(env, nb_episodes = 20, visualize = False)
    print(np.mean(scores.history['episode_reward']))
    return scores.history['episode_reward']

####################################################################################################################################


####################################################################################################################################
# Print for options and arguments
def printHelpAndExit():
    print("the ip address and port are for the remote receiver\n")
    print("the method(m) determines whether to use the RL agent or the clustering technique")
    print("the pid gives the participant ID")
    print("method=<method> (AR or RL or CLUSTER or RL_CLUSTER)")
    print("<script_name>.py  -i <ipaddress> -p <port number> -m <method> --pid <participant ID>\n")
    print("<script_name>.py --ip=<ipaddress> --port=<port number> --method=<method> --pid <participant ID>\n")
    print("<script_name>.py -h")
    print("for help\n")
    print("<script_name>.py -l --- DEPRECATED")
    print("for ip = 0.0.0.0 with port=12344 and method = RL\n")
    sys.exit(2)

def getNormalizedScores(r_scores,ddpg_scores):
    all_scores = r_scores + ddpg_scores

    minimum = min(all_scores)
    maximum = max(all_scores)

    # Max and min possible rewards cannot be easily calculated as it would involve checking all possible select and non-select combinations along with different states
    range = maximum-minimum

    normalized_r_scores = [(x-minimum)/range for x in r_scores]
    normalized_ddpg_scores = [(x-minimum)/range for x in ddpg_scores]

    return normalized_r_scores,normalized_ddpg_scores


def main(argv):
    # referring to global names within the main function
    global method

    global MEASURES_FILE

    global pid

    global started

    # try:
    #     opts,args = getopt.getopt(argv,"hl:m:",["method="])
    # except getopt.GetoptError:
    #     printHelpAndExit()
    #
    # #exit if no arguments
    # if(not opts):
    #     printHelpAndExit()
    #
    # for opt,val in opts:
    #     if(opt == "-h"):
    #         printHelpAndExit()
    #     if(opt == "-l"):
    #         break
    #     if(opt in ("-m","--method")):
    #         method = val





    #instantiate TextProcessor
    textProcessor = TextProcessor()

    #returns the dictionary of postIt notes
    # the ldaModel for identifying topics for the postIt notes
    # the ldaDictionary for identifying topics for the postIt notes
    # postIts, ldaModel, ldaDictionary = textProcessor.postItsFromFile(FILENAMES, NUM_TOPICS)

    # if using BERT
    postIts = textProcessor.postItsFromFile_bert(FILENAMES)

    # if the number of post-its exceed the input size of the NN - this will break:
    if (len(postIts) > NN_INPUT_ROW):
        print("#post-its ("+ str(len(postIts)) + ") > " + "#NN_input ("+ NN_INPUT_ROW +")")


    print("###################################################################################################")
    print("METHOD = "+method)
    print("###################################################################################################")
    ###################################################################################################
    # For RL and RL+CLUSTER
    ###################################################################################################
    if(method == "RL" or method == "RL_CLUSTER"):
        #set up training env
        trainingEnv = SetUpTrainingEnv(postIts,textProcessor)

        #build RL model
        # model = buildDQNModel(trainingEnv)
        actor, critic, critic_action_input = buildDDPGModel(trainingEnv)

        # file to store weights after training
        weight_file ='ddpg_weights.h5f'

        # build the agent
        agent = buildDDPGAgent(actor, critic, critic_action_input, trainingEnv)


        #if weights trained on participants already exists - load them
        #only use when not training agent with simulation
        # loadWeights(agent, weight_file)

        #train RL model
        #if weight_file exists, it will just reload the old weights
        agent = trainDDPGAgent(agent, trainingEnv, weight_file)

        #test using RandomSolver
        r_scores = RandomSolver.solve(trainingEnv, episodes=20)

        #test using agent
        ddpg_scores = testAgent(agent,trainingEnv)

        normalized_r_score, normalized_ddpg_score = getNormalizedScores(r_scores = r_scores,ddpg_scores = ddpg_scores)

        print("Normalized Random Solver Scores are:")
        print(normalized_r_score)

        print("Normalized DDPG Scores are:")
        print(normalized_ddpg_score)

if __name__ == '__main__':
    main(sys.argv[1:])
