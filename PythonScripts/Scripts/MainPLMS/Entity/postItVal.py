class PostItVal(object):
    def __init__(self, id, isSelected, currentPostItState, dwellTimeRatio, saccadeInRatio):
        self.id = id
        self.isSelected = isSelected
        self.currentPostItState = currentPostItState
        self.dwellTimeRatio = dwellTimeRatio
        self.saccadeInRatio = saccadeInRatio
        # need saccade in data

    @classmethod
    def from_json(cls, data):
        return cls(**data)
