import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Lambda, Input
from tensorflow.keras.layers.experimental.preprocessing import Discretization
from tensorflow.keras.optimizers import Adam

import gym
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from keras.layers import concatenate
from keras.models import Model

from functools import reduce

# creates and returns the actor model
class DDPGLearning(object):


##################################################################################
# NOTE: Observations are taken as (batch_size, dim1, dim2...dimN, channel)
##################################################################################

##################################################################################
# The next few networks are for outputting actions
# output layer = number of clues
##################################################################################

##################################################################################
# Dense layer first creates a dense layer with 60 outputs for each post it
# https://stackoverflow.com/questions/53670332/why-not-use-flatten-followed-by-a-dense-layer-instead-of-timedistributed
# but instead of Time Distributed this is Post It Distributed
##################################################################################
    @classmethod
    def create_actor_model(cls, observation_space, action_space):
        input_layer = Input(shape = (1,) + observation_space.shape)
        h0 = Dense(60, activation = 'relu')(input_layer)
        flatten_layer = Flatten()(h0)
        h1 = Dense(24, activation = 'relu')(flatten_layer)
        h2 = Dense(48, activation = 'relu')(h1)
        h3  = Dense(24, activation = 'relu')(h2)
        output = Dense(action_space.shape[0], activation = 'sigmoid')(h3)

        # according to https://keras.io/examples/rl/ddpg_pendulum/
        # we can set the upper bound of our output to match our action space by the following
        # output = output*2 #where three is the number of states.
        output = output*3  # so [0,3] and then we use math.floor

        actor = Model(inputs = input_layer, outputs = output)

        return actor


    # creates the critic model and returns both the model and the action_input_layer
    @classmethod
    def create_critic_model(cls, observation_space, action_space):
        action_input_layer = Input(shape = (action_space.shape[0],), name = 'action_input')
        action_h0 = Dense(24, activation = 'relu')(action_input_layer)

        observation_input_layer = Input(shape = (1,) + observation_space.shape, name = 'observation_input')
        observation_h0 = Dense(60, activation = 'relu')(observation_input_layer)
        flatten_observation = Flatten()(observation_h0)
        observation_h1 = Dense(48, activation = 'relu')(flatten_observation)
        observation_h2 = Dense(24, activation = 'relu')(observation_h1)

        # can only concatenate when the outputs are the same size: in this case 24
        # merged = concatenate([action_input_layer, flatten_observation])
        merged = concatenate([action_h0, observation_h2])
        merged_h0 = Dense(24, activation = 'relu')(merged)
        output = Dense(1, activation = 'relu')(merged_h0)

        critic = Model(inputs = [observation_input_layer,action_input_layer], outputs = output)

        return critic, action_input_layer

##################################################################################
# Dense only layers with SOFTMAX
# WORKS WELL
# cannot use argmax here as it is not differentiable
##################################################################################
    @classmethod
    def create_actor_model_softmax(cls, observation_space, action_space):
        input_layer = Input(shape = (1,) + observation_space.shape)
        h0 = Dense(1000, activation = 'relu')(input_layer)
        h1 = Dense(124, activation = 'relu')(h0)
        h2 = Dense(48, activation = 'relu')(h1)
        h3 = Dense(124, activation = 'relu')(h2)
        h4  = Dense(24, activation = 'relu')(h3)
        output = Dense(action_space.shape[1], activation = 'softmax')(h4) #index 1 of action space = number of states
        output = Flatten()(output)
        # output = tf.math.argmax(output, axis = -1)
        # output = Flatten()(output)
        # output = tf.cast(output,tf.float32)

        actor = Model(inputs = input_layer, outputs = output)

        return actor


    # creates the critic model and returns both the model and the action_input_layer
    @classmethod
    def create_critic_model_softmax(cls, observation_space, action_space):
        action_input_layer = Input(shape = (reduce(lambda x, y: x*y, action_space.shape), ), name = 'action_input')
        flatten_action = Flatten()(action_input_layer)
        action_h0 = Dense(24, activation = 'relu')(flatten_action)

        observation_input_layer = Input(shape = (1,) + observation_space.shape, name = 'observation_input')
        observation_h0 = Dense(60, activation = 'relu')(observation_input_layer)
        flatten_observation = Flatten()(observation_h0)
        observation_h1 = Dense(48, activation = 'relu')(flatten_observation)
        observation_h2 = Dense(24, activation = 'relu')(observation_h1)

        # can only concatenate when the outputs are the same size: in this case 24
        merged = concatenate([action_h0, observation_h2])
        merged_h0 = Dense(24, activation = 'relu')(merged)
        output = Dense(1, activation = 'relu')(merged_h0)

        critic = Model(inputs = [observation_input_layer,action_input_layer], outputs = output)

        return critic, action_input_layer

##################################################################################
# Dense First with Discretization
# DOES NOT WORK
# Discretization function is not differentiable
##################################################################################
    @classmethod
    def create_actor_model_discrete(cls, observation_space, action_space):
        input_layer = Input(shape = (1,) + observation_space.shape)
        h0 = Dense(60, activation = 'relu')(input_layer)
        flatten_layer = Flatten()(h0)
        h1 = Dense(24, activation = 'relu')(flatten_layer)
        h2 = Dense(48, activation = 'relu')(h1)
        h3  = Dense(24, activation = 'relu')(h2)
        h4 = Dense(action_space.shape[0], activation = 'sigmoid')(h3)
        output = h4*2
        output = Discretization(bins = [1., 2.])(output)
        output = tf.cast(output,tf.float32)

        # according to https://keras.io/examples/rl/ddpg_pendulum/
        # we can set the upper bound of our output to match our action space by the following
        # output = output*2 #where three is the number of states.
        # output = output*2  # so [0,3] and then we use math.floor

        actor = Model(inputs = input_layer, outputs = output)

        return actor


    # creates the critic model and returns both the model and the action_input_layer
    @classmethod
    def create_critic_model_discrete(cls, observation_space, action_space):
        action_input_layer = Input(shape = (action_space.shape[0],), name = 'action_input')
        action_h0 = Dense(24, activation = 'relu')(action_input_layer)

        observation_input_layer = Input(shape = (1,) + observation_space.shape, name = 'observation_input')
        observation_h0 = Dense(60, activation = 'relu')(observation_input_layer)
        flatten_observation = Flatten()(observation_h0)
        observation_h1 = Dense(48, activation = 'relu')(flatten_observation)
        observation_h2 = Dense(24, activation = 'relu')(observation_h1)

        # can only concatenate when the outputs are the same size: in this case 24
        merged = concatenate([action_h0, observation_h2])
        merged_h0 = Dense(24, activation = 'relu')(merged)
        output = Dense(1, activation = 'relu')(merged_h0)

        critic = Model(inputs = [observation_input_layer,action_input_layer], outputs = output)

        return critic, action_input_layer

####################################################################
# only dense layers
##################################################################################
    @classmethod
    def create_actor_model_flatEnd(cls, observation_space, action_space):
        input_layer = Input(shape = (1,) + observation_space.shape)
        h0 = Dense(60, activation = 'relu')(input_layer)
        h1 = Dense(24, activation = 'relu')(h0)
        h2 = Dense(48, activation = 'relu')(h1)
        h3  = Dense(24, activation = 'relu')(h2)
        output = Dense(1, activation = 'sigmoid')(h3)
        output = Flatten()(output)

        # according to https://keras.io/examples/rl/ddpg_pendulum/
        # we can set the upper bound of our output to match our action space by the following
        # output = output*2 #where three is the number of states.
        output = output*3  # so [0,3] and then we use math.floor

        actor = Model(inputs = input_layer, outputs = output)

        return actor


    # creates the critic model and returns both the model and the action_input_layer
    @classmethod
    def create_critic_model_flatEnd(cls, observation_space, action_space):
        action_input_layer = Input(shape = (action_space.shape[0],), name = 'action_input')
        action_h0 = Dense(24, activation = 'relu')(action_input_layer)

        observation_input_layer = Input(shape = (1,) + observation_space.shape, name = 'observation_input')
        observation_h0 = Dense(60, activation = 'relu')(observation_input_layer)
        flatten_observation = Flatten()(observation_h0)
        observation_h1 = Dense(48, activation = 'relu')(flatten_observation)
        observation_h2 = Dense(24, activation = 'relu')(observation_h1)

        # can only concatenate when the outputs are the same size: in this case 24
        merged = concatenate([action_h0, observation_h2])
        merged_h0 = Dense(24, activation = 'relu')(merged)
        output = Dense(1, activation = 'relu')(merged_h0)

        critic = Model(inputs = [observation_input_layer,action_input_layer], outputs = output)

        return critic, action_input_layer

##################################################################################
# Flattens the input first resulting in a dense network considering every input as a parameter
# https://stackoverflow.com/questions/53670332/why-not-use-flatten-followed-by-a-dense-layer-instead-of-timedistributed
# but instead of Time Distributed this is Post It Distributed
##################################################################################
    @classmethod
    def create_actor_model_flatFirst(cls, observation_space, action_space):
        input_layer = Input(shape = (1,) + observation_space.shape)
        flatten_layer = Flatten()(input_layer)
        h0 = Dense(24, activation = 'relu')(flatten_layer)
        h1 = Dense(48, activation = 'relu')(h0)
        h2  = Dense(24, activation = 'relu')(h1)
        output = Dense(action_space.shape[0], activation = 'tanh')(h2)

        # according to https://keras.io/examples/rl/ddpg_pendulum/
        # we can set the upper bound of our output to match our action space by the following
        output = output*2 #where three is the number of states.

        actor = Model(inputs = input_layer, outputs = output)

        return actor


    # creates the critic model and returns both the model and the action_input_layer
    @classmethod
    def create_critic_model_flatFirst(cls, observation_space, action_space):
        action_input_layer = Input(shape = (action_space.shape[0],), name = 'action_input')
        action_h0 = Dense(24, activation = 'relu')(action_input_layer)

        observation_input_layer = Input(shape = (1,) + observation_space.shape, name = 'observation_input')
        flatten_observation = Flatten()(observation_input_layer)
        observation_h0 = Dense(48, activation = 'relu')(flatten_observation)
        observation_h1 = Dense(24, activation = 'relu')(observation_h0)

        # can only concatenate when the outputs are the same size: in this case 24
        merged = concatenate([action_h0, observation_h2])
        merged_h0 = Dense(24, activation = 'relu')(merged)
        output = Dense(1, activation = 'relu')(merged_h0)

        critic = Model(inputs = [observation_input_layer,action_input_layer], outputs = output)

        return critic, action_input_layer

##################################################################################
# Network for threshold outputs
# output only has two nodes - low threshold and high threshold
##################################################################################
    @classmethod
    def create_actor_model_threshold(cls, observation_space, action_space):
        input_layer = Input(shape = (1,) + observation_space.shape)
        h0 = Dense(60, activation = 'relu')(input_layer)
        flatten_layer = Flatten()(h0)
        h1 = Dense(24, activation = 'relu')(flatten_layer)
        h2 = Dense(48, activation = 'relu')(h1)
        h3  = Dense(24, activation = 'relu')(h2)
        #sigmoid as putput space is in between 0 and 1
        output = Dense(3, activation = 'sigmoid')(h3)

        actor = Model(inputs = input_layer, outputs = output)

        return actor


    # creates the critic model and returns both the model and the action_input_layer
    @classmethod
    def create_critic_model_threshold(cls, observation_space, action_space):
        action_input_layer = Input(shape = (3,), name = 'action_input')
        action_h0 = Dense(24, activation = 'relu')(action_input_layer)

        observation_input_layer = Input(shape = (1,) + observation_space.shape, name = 'observation_input')
        observation_h0 = Dense(60, activation = 'relu')(observation_input_layer)
        flatten_observation = Flatten()(observation_h0)
        observation_h1 = Dense(48, activation = 'relu')(flatten_observation)
        observation_h2 = Dense(24, activation = 'relu')(observation_h1)

        # can only concatenate when the outputs are the same size: in this case 24
        merged = concatenate([action_h0, observation_h2])
        merged_h0 = Dense(24, activation = 'relu')(merged)
        output = Dense(1, activation = 'relu')(merged_h0)

        critic = Model(inputs = [observation_input_layer,action_input_layer], outputs = output)

        return critic, action_input_layer


##################################################################################
# GENERAL METHODS
##################################################################################

    @classmethod
    def build_model(cls, observation_space, action_space):
        actor = cls.create_actor_model(observation_space, action_space)
        critic, critic_action_input = cls.create_critic_model(observation_space, action_space)
        return actor, critic, critic_action_input

    @classmethod
    def build_model_softmax(cls, observation_space, action_space):
        actor = cls.create_actor_model_softmax(observation_space, action_space)
        critic, critic_action_input = cls.create_critic_model_softmax(observation_space, action_space)
        return actor, critic, critic_action_input

    @classmethod
    def build_model_discrete(cls, observation_space, action_space):
        actor = cls.create_actor_model_discrete(observation_space, action_space)
        critic, critic_action_input = cls.create_critic_model_discrete(observation_space, action_space)
        return actor, critic, critic_action_input

    @classmethod
    def build_model_flatEnd(cls, observation_space, action_space):
        actor = cls.create_actor_model_flatEnd(observation_space, action_space)
        critic, critic_action_input = cls.create_critic_model_flatEnd(observation_space, action_space)
        return actor, critic, critic_action_input

    @classmethod
    def build_model_flatFirst(cls, observation_space, action_space):
        actor = cls.create_actor_model_flatFirst(observation_space, action_space)
        critic, critic_action_input = cls.create_critic_model_flatFirst(observation_space, action_space)
        return actor, critic, critic_action_input

    @classmethod
    def build_model_threshold(cls, observation_space, action_space):
        actor = cls.create_actor_model_threshold(observation_space, action_space)
        critic, critic_action_input = cls.create_critic_model_threshold(observation_space, action_space)
        return actor, critic, critic_action_input

    @classmethod
    def build_agent(cls, actorModel, criticModel, critic_action_input_layer, action_space):
        # for all others
        # actions = action_space.shape[0]
        #for SOFTMAX
        # actions = action_space.shape[0]*action_space.shape[1]

        # # DOES NOT WORK as in ddpg.py of keras core - (WHY?!)
        # # line 184 flattens for actor
        # # line 261 does not flatten for target actor
        # actions = action_space.shape

        # seems to work for both
        # basically finds the size of the flattened action space
        actions = reduce(lambda x, y: x*y, action_space.shape)
        memory = SequentialMemory(limit = 100000, window_length = 1)
        random_process = OrnsteinUhlenbeckProcess(size = actions, theta = .15, mu = 0.,sigma = 0.3)

        ddpg = DDPGAgent(
                            nb_actions = actions,
                            actor = actorModel,
                            critic = criticModel,
                            critic_action_input = critic_action_input_layer,
                            memory = memory,
                            gamma = 0.95, # DISCOUNT FACTOR - future rewards may not be as predictable
                            nb_steps_warmup_critic = 100,
                            nb_steps_warmup_actor = 100,
                            random_process = random_process,
                            target_model_update = 1e-3
                        )
        return ddpg
