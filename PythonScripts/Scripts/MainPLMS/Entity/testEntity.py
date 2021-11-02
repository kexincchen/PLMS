class TestEntity(object):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    @classmethod
    def from_json(cls, data):
        return cls(**data)


#for msgpack but unusable currently
def decode(obj):
    if isinstance(obj,dict):
        return TestEntity(obj['key'],obj['value'])
    return obj

def encode(obj):
    if isinstance(obj,TestEntity):
        return {
            "key":obj.key,
            "value": obj.value
            }
    return obj
