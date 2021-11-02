import sys
sys.path.append("..")

from Entity.postIt import PostIt

class TextProcessor(object):

    @classmethod
    def postItsFromFile(self,filename):
        file =  open(filename,"r")

        id = 0 #counter for the # of postit notes
        postItDict = {}
        for line in file:
            print line
            note = PostIt()
            note.id = id
            note.clue = line
            #note.header = getHeader(line)
            #note.topics = NLPTopicIdentify(line)
            postItDict[note.id] = note
            id = id+1

        file.close()
        return postItDict
