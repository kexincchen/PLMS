class ActionMap(object):
    def __init__(self, id, action = 1):
        self.id = id
        self.action = action

    @classmethod
    def from_json(cls, data):
        return cls(**data)
