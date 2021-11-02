import sys,getopt
import time
import os.path
import numpy as np
import gym
import math

from tensorflow.keras.optimizers import Adam

from Network.udp import UDPSocket
from Entity.jsonMessage import JsonMessage
from Entity.postItContent import PostItContent
from Entity.postItNumber import PostItNumber
from Entity.postItData import PostItData
from Entity.postItVal import PostItVal
from Entity.postItValList import PostItValList
from Entity.actionMap import ActionMap
from Entity.actionMapList import ActionMapList
from Entity.completionTime import CompletionTime

from TextProcessor.textProcessor import TextProcessor

from RL.Environments.sensemakingTrainingEnv import SensemakingTrainingEnv
from RL.Environments.sensemakingTrainingEnv2 import SensemakingTrainingEnv2
from RL.Environments.eyeAndStateTrainingEnv import EyeAndStateTrainingEnv
from RL.Environments.eyeAndStateEnv import EyeAndStateEnv
from RL.Environments.eyeAndStateThreshTrainingEnv import EyeAndStateThreshTrainingEnv
from RL.Environments.eyeAndStateThreshEnv import EyeAndStateThreshEnv
from RL.Environments.bertTrainingEnv import BertTrainingEnv
from RL.Environments.bertEnv import BertEnv

from RL.Solvers.randomSolver import RandomSolver
from RL.Solvers.deepQLearning import DeepQLearning
from RL.Solvers.ddpgLearning import DDPGLearning
from RL.Solvers.clustering import Clustering

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

# maximum number of postits that can be selected per step by the dummy user
MAX_POST_PER_STEP = 3

####################################################################################################################################

####### sets up the ip, port of the udp socket for sending
####### binds to listen values
### list is called after
# static listens
udp_ip_listen = "0.0.0.0"
udp_port_listen = 12344

udp_ip_remote = "192.168.137.1"
udp_port_remote = 12344

#method to use: AR or RL or CLUSTERING or RL_CLUSTER
method = "RL"

##relevant files

#files that stores participant measures
data_file_path = "../../Data/Participant_Data/"

MEASURES_FILE = "participant_measures.txt"

#stanford murder mystery
filename1 = "../../Data/murder_mystery/stanford_mm.txt"
#stanford murder mystery DISTRACTIONS
filename2 = "../../Data/murder_mystery/stanford_mm_distractions.txt"

#bank robbery mystery
# filename1 = "../../Data/murder_mystery/bank_robbery.txt"

FILENAMES = []
FILENAMES.append(filename1)
FILENAMES.append(filename2)

# for storing the participant ID
pid = -1


# #TESTING some functions
# textProcessor.test(filenames, ldaModel, ldaDictionary)
# print(SensemakingTrainingEnv.getDict(postIts)
# print(len(postIts))
# print(SensemakingTrainingEnv.getBox(postIts,NUM_TOPICS))

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
def SetUpDeploymentEnv(postIts, udpSocket, udp_ip_remote, udp_port_remote):
    # env = EyeAndStateEnv(
    #                         len(postIts),
    #                         NUM_TOPICS,
    #                         NUM_STATES,
    #                         EyeAndStateEnv.getBox(postIts,NUM_TOPICS),
    #                         udpSocket,
    #                         udp_ip_remote,
    #                         udp_port_remote
    #                     )

    # env = EyeAndStateThreshEnv(
    #                         len(postIts),
    #                         NUM_TOPICS,
    #                         NUM_STATES,
    #                         EyeAndStateEnv.getBox(postIts,NUM_TOPICS),
    #                         udpSocket,
    #                         udp_ip_remote,
    #                         udp_port_remote
    #                     )

    ######################################################
    # BERT SENTENCE EMBEDDINGS
    ######################################################

    env = BertEnv(
                            len(postIts),
                            NN_INPUT_ROW,
                            EMBEDDING_SIZE,
                            NUM_STATES,
                            BertEnv.getBox(postIts, EMBEDDING_SIZE, NN_INPUT_ROW),
                            udpSocket,
                            udp_ip_remote,
                            udp_port_remote
                        )
    return env

####################################################################################################################################

#using a random solver -> picks random actions
# RandomSolver.solve(env)

####################################################################################################################################
#build model
def buildDQNModel(env):
    deepQModel = DeepQLearning.build_model(
                            env.observation_space,
                            env.action_space
                        )
    deepQModel.summary()
    return deepQModel

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

####################################################################################################################################
#train Agent
def trainDQNAgent(model, env, weight_file):
    agent = DeepQLearning.build_agent(model, env.action_space)
    agent.compile(Adam(lr = 1e-3), metrics = ['mae'])

    if(os.path.exists(weight_file + '.index') and (os.path.exists(weight_file + ".data-00000-of-00001"))):
        # print("file exists")
        agent.load_weights(weight_file)
    else:
        # print("file does not exists")
        agent.fit(env, nb_steps = 50000, visualize = False, verbose = 1)
        agent.save_weights(weight_file, overwrite = True)

    return agent

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
    else:
        # print("file does not exists")
        agent.fit(env, nb_episodes = 138, visualize = False, verbose = 1)
        # agent.fit(env, nb_episodes = 15, visualize = False, verbose = 1) # for TESTING
        agent.save_weights(weight_file, overwrite = True)

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
    scores = agent.test(env, nb_episodes = 5, visualize = False)
    print(np.mean(scores.history['episode_reward']))

####################################################################################################################################

#Code for handling message
# takes socket for communication
# takes the msg to handle
# takes the dictionary of postIts to send to the HoloLens
# takes the trained agent to make predictions and send actions to HoloLens
def handleMsgJson(udpSocket, msg, postIts):
    jm = JsonMessage.from_json(msg)
    # print("In Main, type = "+jm.messageType)
    #do something based on the message
    if(jm.messageType == "Start"):
        print("*************** Started ***************")
        udpSocket.setStarted(True)

        #send the post-it notes
        for pc in postIts:
            jm = JsonMessage("PostItContent",postIts[pc].postItContent)
            udpSocket.sendUDPMsg(jm, udp_ip_remote, udp_port_remote)

        #send number of post-its or clues
        time.sleep(2.5)

        print("Number of post-its = "+str(len(postIts)))
        jm = JsonMessage("PostItNumber", PostItNumber(len(postIts)))
        udpSocket.sendUDPMsg(jm, udp_ip_remote, udp_port_remote)

    # this does not seem to work as the message is too large
    # received msg seems to be broken into parts
    # does not receive the whole message from the HoloLens
    if(jm.messageType == "PostItValList"):
        pValList = jm.messageObject
        for pVal in pValList.values:
            postIts[pVal.id].state = pVal.currentPostItState
            postIts[pVal.id].isSelected = pVal.isSelected
            postIts[pVal.id].lastEyeDwellRatio = pVal.dwellTimeRatio
            postIts[pVal.id].saccadeInRatio = pVal.saccadeInRatio

        obs_space = udpSocket.envClass.getBox(postIts, udpSocket.emb_size, udpSocket.nn_input_rows)

        udpSocket.writeObs(obs_space, EMBEDDING_SIZE)

        if(udpSocket.methodName == "RL" or udpSocket.methodName == "RL_CLUSTER"):
            udpSocket.obsQueue.append(obs_space)

    if(jm.messageType == "PostItVal"):
        udpSocket.pValList.values.append(jm.messageObject)

        if(len(udpSocket.pValList.values) == len(postIts)):
            for pVal in udpSocket.pValList.values:
                postIts[pVal.id].state = pVal.currentPostItState
                postIts[pVal.id].isSelected = pVal.isSelected
                postIts[pVal.id].lastEyeDwellRatio = pVal.dwellTimeRatio
                postIts[pVal.id].saccadeInRatio = pVal.saccadeInRatio


            obs_space = udpSocket.envClass.getBox(postIts, udpSocket.emb_size, udpSocket.nn_input_rows)

            udpSocket.writeObs(obs_space, EMBEDDING_SIZE)

            udpSocket.pValList.values.clear()

            if(udpSocket.methodName == "RL" or udpSocket.methodName == "RL_CLUSTER"):
                udpSocket.obsQueue.append(obs_space)


    if(jm.messageType == "WhiteBoardPostItMapList"):
        # perform clustering on jm.messageObject
        # Clustering.clusterPostIts(jm.messageObject, postIts, NUM_TOPICS) # for LDA
        clusteredPostIts = Clustering.clusterPostIts(jm.messageObject, postIts, EMBEDDING_SIZE) # for bert

        #if no selections have been made, there will be no clusters
        if len(clusteredPostIts.values) == 0:
            return

        if(udpSocket.methodName == "CLUSTER" or udpSocket.methodName == "RL_CLUSTER"):
            # send clustered WhiteBoardPostItMapList
            jm = JsonMessage("WhiteBoardPostItMapList", clusteredPostIts)
            udpSocket.sendUDPMsg(jm, udp_ip_remote, udp_port_remote)

        udpSocket.writeClusters(clusteredPostIts)

    if(jm.messageType == "WhiteBoardPostItMap"):
        udpSocket.wbPostItList.values.append(jm.messageObject)

        if(len(udpSocket.wbPostItList.values) == len(postIts)):
            # perform clustering
            # Clustering.clusterPostIts(udpSocket.wbPostItList, postIts, NUM_TOPICS) # for LDA
            clusteredPostIts = Clustering.clusterPostIts(udpSocket.wbPostItList, postIts, EMBEDDING_SIZE) # for bert

            #if no selections have been made, there will be no clusters
            if len(clusteredPostIts.values) == 0:
                udpSocket.wbPostItList.values.clear()
                return

            if(udpSocket.methodName == "CLUSTER" or udpSocket.methodName == "RL_CLUSTER"):
                # send clustered WhiteBoardPostItMapList
                for wbPostItMap in clusteredPostIts.values:
                    jm = JsonMessage("WhiteBoardPostItMap",wbPostItMap)
                    udpSocket.sendUDPMsg(jm, udp_ip_remote, udp_port_remote)

            udpSocket.wbPostItList.values.clear()

            udpSocket.writeClusters(clusteredPostIts)


    if(jm.messageType == "Completed"):
        if(udpSocket.isEpisodeDone == False):
            print("*************** Finished ***************")
            udpSocket.isEpisodeDone = True
            file =  open(udpSocket.data_file_path + udpSocket.record_file_name, "a+")
            file.write(str(udpSocket.pid) + "\t" + udpSocket.methodName + "\t" + str(jm.messageObject.completionTime))
            file.write("\n")
            file.close()

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


def main(argv):
    # referring to global names within the main function
    global udp_ip_remote
    global udp_port_remote

    global method

    global MEASURES_FILE

    global pid

    global started

    try:
        opts,args = getopt.getopt(argv,"hli:p:m:",["ip=","port=","method=","pid="])
    except getopt.GetoptError:
        printHelpAndExit()

    #exit if no arguments
    if(not opts):
        printHelpAndExit()

    for opt,val in opts:
        if(opt == "-h"):
            printHelpAndExit()
        if(opt == "-l"):
            break
        if(opt in ("-i","--ip")):
            udp_ip_remote = val
        if(opt in ("-p","--port")):
            udp_port_remote = int(val)
        if(opt in ("-m","--method")):
            method = val
        if(opt in ("--pid",)):
             pid = int(val)


    udpSocket = UDPSocket()
    udpSocket.initSocket() #initialize socket
    udpSocket.bindSock(udp_ip_listen,udp_port_listen) # bind socket to listen on predefined ip and port

    #sets the file name for the udpsocket to
    udpSocket.setRecordFileName(MEASURES_FILE)

    #  set data file path
    udpSocket.setDataFilePath(data_file_path)

    #set pid
    if pid == -1:
        print("Please set the participant ID using --pid=<participant ID>")
        sys.exit(1)
    udpSocket.setPID(pid)

    udpSocket.listenOnThread(handleMsgJson)

    #instantiate TextProcessor
    textProcessor = TextProcessor();

    #returns the dictionary of postIt notes
    # the ldaModel for identifying topics for the postIt notes
    # the ldaDictionary for identifying topics for the postIt notes
    # postIts, ldaModel, ldaDictionary = textProcessor.postItsFromFile(FILENAMES, NUM_TOPICS)

    # if using BERT
    postIts = textProcessor.postItsFromFile_bert(FILENAMES)

    # if the number of post-its exceed the input size of the NN - this will break:
    if (len(postIts) > NN_INPUT_ROW):
        print("#post-its ("+ str(len(postIts)) + ") > " + "#NN_input ("+ NN_INPUT_ROW +")")

    #udp set method name
    udpSocket.setMethodName(method)

    #set size of nn to determine observation_space
    udpSocket.setNNInputRowSize(NN_INPUT_ROW)

    #set up postits to be sent
    udpSocket.setPostIts(postIts)

    print("###################################################################################################")
    print("METHOD = "+method)
    print("###################################################################################################")

    #set the environment class to observation Box
    # this is needed to process the messages received from the holoLens even when not using the agent
    # even if the environment class and embedding size is related to the agent

    # udpSocket.setEnvClass(EyeAndStateEnv)
    # udpSocket.setEmbSize(NUM_TOPICS)
    udpSocket.setEnvClass(BertEnv)
    udpSocket.setEmbSize(EMBEDDING_SIZE)

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
        # RandomSolver.solve(trainingEnv)

        #test using agent
        # testAgent(agent,trainingEnv)

        # TESTING
        # sending PostItVal and postItValList
        # pVal1 = PostItVal(1,False,1,0.46)
        # pVal2 = PostItVal(13,True,2,0.90)
        # pList = PostItValList([pVal1,pVal2])
        # jm = JsonMessage("PostItValList",pList)
        # udpSocket.sendUDPMsg(jm, udp_ip_remote, udp_port_remote)
        # END TESTING

        deployEnv = SetUpDeploymentEnv(postIts, udpSocket, udp_ip_remote, udp_port_remote)

        # this does not look thread safe
        while(not udpSocket.started):
            pass

        agent.fit(deployEnv, nb_episodes = 1, visualize = False, verbose = 1)

        # if we plan to save the adjusted weights from participants
        # agent.save_weights(weight_file, overwrite = True)


    ##############################################################
    # can use to predict actions for single state
    ##############################################################

    # sample = trainingEnv.observation_space.sample()
    # sample[:,NUM_TOPICS] = np.round(sample[:,NUM_TOPICS])
    # sample[:,NUM_TOPICS+1] = np.round(sample[:,NUM_TOPICS+1])
    # print(sample)
    #
    # lower_bound_action = 0
    # upper_bound_action = 2
    # action = actor.predict(sample.reshape((1,1) + sample.shape))
    # action = np.clip(action, lower_bound_action, upper_bound_action)
    # print(action)

    ###################################################################################################
    # End For RL
    ###################################################################################################

    while(not udpSocket.isEpisodeDone):
        time.sleep(1)


if __name__ == '__main__':
    main(sys.argv[1:])
