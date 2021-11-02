from .actionMap import ActionMap

class ActionMapList(object):
    def __init__(self, values = None):
        self.values = values
        if self.values == None:
            self.values = []

    @classmethod
    def from_json(cls, data):
        values = []
        for val in data["values"]:
            values.append(ActionMap.from_json(val))
        return cls(values)
