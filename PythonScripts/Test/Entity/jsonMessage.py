from testEntity import TestEntity
from postItContent import PostItContent

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
        if(mType == "PostItContent"):
            mObj = PostItContent.from_json(data["messageObject"])

        return cls(mType,mObj)
