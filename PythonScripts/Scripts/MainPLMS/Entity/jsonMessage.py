from .testEntity import TestEntity
from .postItNumber import PostItNumber
from .postItContent import PostItContent
from .postItVal import PostItVal
from .postItValList import PostItValList
from .actionMap import ActionMap
from .actionMapList import ActionMapList
from .completionTime import CompletionTime
from .whiteBoardPostItMap import WhiteBoardPostItMap
from .whiteBoardPostItMapList import WhiteBoardPostItMapList

class JsonMessage(object):
    def __init__(self, messageType, messageObject):
        self.messageType = messageType
        self.messageObject = messageObject

    @classmethod
    def from_json(cls, data):
        mType = data["messageType"]
        mObj = None

        if(mType == "Start"):
            mObj = None
        if(mType == "TestEntity"):
            mObj = TestEntity.from_json(data["messageObject"])
        if(mType == "PostItNumber"):
            mObj == PostItNumber.from_json(data["messageObject"])
        if(mType == "PostItContent"):
            mObj = PostItContent.from_json(data["messageObject"])
        if(mType == "PostItVal"):
            mObj = PostItVal.from_json(data["messageObject"])
        if(mType == "PostItValList"):
            mObj = PostItValList.from_json(data["messageObject"])
        if(mType == "ActionMap"):
            mObj = ActionMap.from_json(data["messageObject"])
        if(mType == "ActionMapList"):
            mObj = ActionMapList.from_json(data["messageObject"])
        if(mType == "WhiteBoardPostItMap"):
            mObj = WhiteBoardPostItMap.from_json(data["messageObject"])
        if(mType == "WhiteBoardPostItMapList"):
            mObj = WhiteBoardPostItMapList.from_json(data["messageObject"])
        if(mType == "Completed"):
            mObj = CompletionTime.from_json(data["messageObject"])

        return cls(mType,mObj)
