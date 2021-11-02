import json
import jsonpickle
import msgpack
import socket
import sys,getopt
import threading
import time
import numpy as np
import os.path

sys.path.append("..")

from Entity.jsonMessage import JsonMessage
from Entity.postItVal import PostItVal
from Entity.postItValList import PostItValList

from Entity.whiteBoardPostItMap import WhiteBoardPostItMap
from Entity.whiteBoardPostItMapList import WhiteBoardPostItMapList

class UDPSocket(object):

    def __init__(self):
        self.obsQueue = []
        self.isEpisodeDone = False
        self.pValList = PostItValList()
        self.wbPostItList = WhiteBoardPostItMapList()
        self.started = False

    #for listening
    def bindSock(self, ip, port):
        self.sock.bind((ip,port))

    def listen(self, handle = None):
        while True:
            data,addr = self.sock.recvfrom(1024)
            # print("received message", data)
            msg = self.deserializeMsgJson(data)
            if(handle == None):
                self.handleMsgJson(msg)
            else:
                handle(self, msg, self.postIts)

    def listenOnThread(self, handle = None):
        listenThread = threading.Thread(target = self.listen, args = (handle,))
        listenThread.start()

    def handleMsgJson(self, msg):
        jm = JsonMessage.from_json(msg)
        print("In udp_class, type = "+jm.messageType)
        #do something based on the message

    def initSocket(self):
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    # sends a msg to the remote (ip,port)
    ## message here is always a jsonMessage.JsonMessage
    def sendUDPMsg(self, msg, remote_ip, remote_port):
        serializedMsg = self.serializeMsgJson(msg)
        self.sock.sendto(serializedMsg.encode('utf-8'),(remote_ip,remote_port))

    def serializeMsgJson(self, msg):
        return jsonpickle.encode(msg)

    def deserializeMsgJson(self, msg):
        return jsonpickle.decode(msg)

####################################################################################################################################
    # the following functions are not related to udp communication
    # they are utility functions to enable the socket on the thread to function
    # this should be sent as parameters in the handleMsg function or put in another helper class.
####################################################################################################################################
    def setPostIts(self, postIts):
        self.postIts = postIts

# to check whether the first message from the holoLens has arrived
# this is to determine when to start sending actions
    def setStarted(self, isStarted):
        self.started = isStarted

    # set the class for which gym.space BOX is needed
    def setEnvClass(self, envClass):
        self.envClass = envClass

    def setEmbSize(self, size):
        #emb_size =  number topics if LDA
        #emb_size = size of sentence embedding vector for BERT -> base = 768
        self.emb_size = size

    def setPID(self, pid):
        self.pid = pid

    def setMethodName(self, methodName):
        self.methodName = methodName

    def setNNInputRowSize(self, nn_input_rows):
        self.nn_input_rows = nn_input_rows

    # set items to write data to files
    def setDataFilePath(self, data_file_path):
        self.data_file_path = data_file_path

    def setRecordFileName(self, record_file_name):
        self.record_file_name = record_file_name

    def writeObs(self, observations, embedding_size):
        #create np indices to store
        indices = range(0, len(self.postIts))
        #convert to vector
        indices = np.array(indices).reshape(len(indices),1)

        obs_to_write = np.append(indices, observations[:len(self.postIts)], axis = 1)

        #save embeddings - only once as this does not change within an episode
        p_obs_embeddings_fileName = self.data_file_path + self.methodName + "/P" + str(self.pid) + "_embeddings.csv"
        if(not os.path.exists(p_obs_embeddings_fileName)):
            f = open(p_obs_embeddings_fileName, "a+", newline = '')
            np.savetxt(f, obs_to_write[:, : (embedding_size+1)], delimiter = ",")
            f.close()

        states_range = [0] + list(range(embedding_size + 1, observations.shape[1] + 1)) # we add 1 since we added the indices
        #save states
        p_obs_states_fileName = self.data_file_path + self.methodName + "/P" + str(self.pid) + "_states.csv"
        f = open(p_obs_states_fileName, "a+", newline = '')
        np.savetxt(f, obs_to_write[:, states_range], delimiter = ",")
        f.write("\n")
        f.close()


    def writeClusters(self, whiteBoardPostItMapList):
        p_cluster_fileName = self.data_file_path + self.methodName + "/P" + str(self.pid) +"_clusters.csv"

        # index of the cluster represents ID of the post-it note
        clusters = []
        for whiteBoardPostItMap in whiteBoardPostItMapList.values:
            clusters.append([whiteBoardPostItMap.postItID, whiteBoardPostItMap.whiteBoardID])

        clusters.sort(key = lambda x: x[0])

        f = open(p_cluster_fileName, "a+", newline = '')
        np.savetxt(f, clusters, delimiter = ",")
        f.write("\n")
        f.close()
