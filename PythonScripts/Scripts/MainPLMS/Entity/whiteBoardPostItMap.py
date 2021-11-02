class WhiteBoardPostItMap(object):
    def __init__(self, whiteBoardID, postItID):
        self.whiteBoardID = whiteBoardID
        self.postItID = postItID

    @classmethod
    def from_json(cls, data):
        return cls(**data)
