from .postItContent import PostItContent
from gym.spaces import Box,Discrete,Dict

class PostItData(object):
    def __init__(self, postItContent = None, topics = None, state = 1, isSelected = False, lastEyeDwellRatio = 0.0, saccadeInRatio = 0.0):
        self.postItContent = postItContent
        self.topics = topics
        self.state = state #[0,1,2] 0 -> Min, 1 -> Max, 2 -> Highlight
        self.isSelected = isSelected
        self.lastEyeDwellRatio = lastEyeDwellRatio
        self.saccadeInRatio = saccadeInRatio
        # need saccade in data
