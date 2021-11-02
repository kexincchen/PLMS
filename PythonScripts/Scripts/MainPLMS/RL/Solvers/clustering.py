import sys
sys.path.append("../..")

import numpy as np
import math
import copy

from Entity.whiteBoardPostItMap import WhiteBoardPostItMap
from Entity.whiteBoardPostItMapList import WhiteBoardPostItMapList
from TextProcessor.textProcessor import TextProcessor

class Clustering(object):
    @classmethod
    def clusterPostIts(cls, wbPostItList, postIts, embedding_size):
        # do not use Mutable defaults in python, they store as static members
        clusteredPostIts = WhiteBoardPostItMapList()

        # get a dictionary of whiteboards to list of attached postit notes
        wbPostItDict = {}
        for wbPostItMap in wbPostItList.values:
            # create the list before we can start appending
            if wbPostItMap.whiteBoardID not in wbPostItDict:
                wbPostItDict[wbPostItMap.whiteBoardID] = []

            wbPostItDict[wbPostItMap.whiteBoardID].append(wbPostItMap.postItID)

        # get averages of sentence emebbinds of post its for each whiteboard
        wbPostItAvg = {}
        for key, value in wbPostItDict.items():
            # no need to calculate avg for non selected post it notes
            if key == -1:
                continue

            count  = 0
            avg = np.zeros(embedding_size) # gets the size of the embedding
            for postItID in value:
                avg = avg + postIts[postItID].topics
                count += 1
            avg = avg/count
            wbPostItAvg[key] = avg


        ##########################################################
        # using Euclidean Distance
        ##########################################################
        # for each post it note that has not been assigned to a whiteboard - denoted by -1
        # for postItID in wbPostItDict[-1]:
        #     min_wbID = -1
        #     min_dist = math.inf
        #     for wbID,avg in wbPostItAvg.items():
        #         if(TextProcessor.euclideanDistance(avg, postIts[postItID].topics) < min_dist):
        #             min_wbID = wbID
        #             min_dist = TextProcessor.euclideanDistance(avg, postIts[postItID].topics)
        #     if(min_wbID != -1): # necesary as wbPostItAvg may be empty
        #         wbPostItMap = WhiteBoardPostItMap(min_wbID, postItID)
        #         clusteredPostIts.values.append(wbPostItMap)

        ##########################################################
        # using Angular Similarity
        ##########################################################
        # for each post it note that has not been assigned to a whiteboard - denoted by -1
        # CHECK IF wbPostItDict[-1] exists
        nonAttachedNotes = wbPostItDict.get(-1,[])
        if(len(nonAttachedNotes) != 0):
            for postItID in nonAttachedNotes:
                max_wbID = -1
                max_sim = -math.inf
                for wbID,avg in wbPostItAvg.items():
                    if(TextProcessor.angular_similarity(avg, postIts[postItID].topics) > max_sim):
                        max_wbID = wbID
                        max_sim = TextProcessor.angular_similarity(avg, postIts[postItID].topics)
                if(max_wbID != -1): # necesary as wbPostItAvg may be empty
                    wbPostItMap = WhiteBoardPostItMap(max_wbID, postItID)
                    clusteredPostIts.values.append(wbPostItMap)

        # add remaining post it notes clustered to their attached whiteboard

        # selected_postItIDs = [postItIDs for wbId, postItIDs in wbPostItDict.items() if wbID not in [-1]]
        # # flatten the above list
        # selected_postItIDs = [postItID for postItIDs in selected_postItIDs for postItID in postItIDs]

        for wbID, postItIDs in wbPostItDict.items():
            if wbID != -1:
                for postItID in postItIDs:
                    wbPostItMap = WhiteBoardPostItMap(wbID, postItID)
                    clusteredPostIts.values.append(wbPostItMap)

        return clusteredPostIts
