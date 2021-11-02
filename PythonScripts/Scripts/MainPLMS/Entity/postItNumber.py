class PostItNumber(object):
    def __init__(self, number):
        self.number = number
        
    @classmethod
    def from_json(cls, data):
        return cls(**data)
