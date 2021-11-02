import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Lambda, Input
from tensorflow.keras.optimizers import Adam
import gym
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.layers import concatenate
from keras.layers import Concatenate
from keras.models import Model

def flattenDict(obs_instance):
    gym.space.flatten(observation_space,obs_instance)

class DeepQLearning(object):

    @classmethod
    #states = number of states after flattened
    #actions = number of actions
    def build_model(cls, observation_space, action_space):
        states = observation_space.shape
        actions = action_space.shape[0]
        model = Sequential()
        # model.add(Flatten(input_shape = (1,states[0],states[1])))
        model.add(Flatten(input_shape = (1,) + states))
        model.add(Dense(24,activation = 'relu'))
        model.add(Dense(24,activation = 'relu'))
        model.add(Dense(actions, activation = 'linear'))
        return model

    # @classmethod
    # def build_model(self, number_clues, number_topics, number_states, action_space):
    #     actions = action_space.shape[0]
    #
    #     model_topic_probs = Sequential()
    #     model_topic_probs.add(Flatten(input_shape = (number_clues,)+(number_topics,), name = 'topic_probs'))
    #     model_topic_input = Input(shape = (number_clues,)+(number_topics,), name = 'topic_probs', dtype = 'float32')
    #     model_topic_encoded = model_topic_probs(model_topic_input)
    #     model_topic_probs.summary()
    #
    #     model_states = Sequential()
    #     model_states.add(Flatten(input_shape = (number_clues,)+(1,), name = 'states'))
    #     model_states_input = Input(shape = (number_clues,)+(1,), name = 'states', dtype = 'int32')
    #     model_states_encoded = model_states(model_states_input)
    #     model_states.summary()
    #
    #     model_isSelected =  Sequential()
    #     model_isSelected.add(Flatten(input_shape = (number_clues,) + (1,), name = 'isSelected'))
    #     model_isSelected_input = Input(shape = (number_clues,) + (1,), name = 'isSelected', dtype = 'int32')
    #     model_isSelected_encoded = model_isSelected(model_isSelected_input)
    #     model_isSelected.summary()
    #
    #     con = Concatenate(axis = -1)([model_topic_encoded, model_states_encoded, model_isSelected_encoded])
    #
    #     hidden = Dense(24,activation = 'relu')(con)
    #     hidden = Dense(24,activation = 'relu')(hidden)
    #
    #     output = Dense(actions, activation = 'linear')(hidden)
    #
    #     model_final = Model(
    #                             inputs = [model_topic_encoded, model_states_encoded, model_isSelected_encoded],
    #                             outputs = output
    #                         )
    #
    #     return model_final

    @classmethod
    def build_agent(cls, deepQModel, action_space):
        actions = action_space.shape[0]
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit = 50000, window_length = 1)
        dqn = DQNAgent(
                        model = deepQModel,
                        memory = memory,
                        policy = policy,
                        nb_actions = actions,
                        nb_steps_warmup = 50,
                        target_model_update = 1e-2
                    )
        return dqn
