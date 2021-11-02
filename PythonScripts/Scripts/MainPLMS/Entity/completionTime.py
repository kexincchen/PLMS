class CompletionTime(object):
    def __init__(self, completionTime):
        self.completionTime = completionTime

    @classmethod
    def from_json(cls, data):
        return cls(**data)
