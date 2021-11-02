class PostItContent(object):
    def __init__(self, id = 0, clue = None, header = None, topics = None):
        self.id = id
        self.clue = clue
        self.header = header
        self.topics = topics

    @classmethod
    def from_json(cls, data):
        return cls(**data)
