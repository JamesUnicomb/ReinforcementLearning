import gym

import os, inspect

import numpy as np
from math import *

from sklearn.utils import shuffle

import theano
import theano.tensor as T
from theano.tensor.extra_ops import to_one_hot
from theano.tensor.raw_random import multinomial

import lasagne
from lasagne.updates import adam, norm_constraint
from lasagne.objectives import squared_error
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, Conv2DLayer, FlattenLayer, DimshuffleLayer, \
                           get_output, get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify, softmax, tanh, linear
from lasagne.init import Constant, Normal


class DeepQNetwork:
    def __init__(self,
                 atari_env,
                 state_dimension,
                 action_dimension,
                 monitor_env   = False,
                 learning_rate = 0.001,
                 critic_update = 10,
                 train_step    = 1,
                 gamma         = 0.95,
                 eps_max       = 1.0,
                 eps_min       = 0.1,
                 eps_decay     = 10000,
                 n_epochs      = 10000,
                 batch_size    = 32,
                 buffer_size   = 50000):

        self.env = gym.make(atari_env)
        if monitor_env:
            None

        self.state_dimension  = state_dimension
        self.action_dimension = action_dimension
        self.learning_rate    = learning_rate
        self.critic_update    = critic_update
        self.train_step       = train_step
        self.gamma            = gamma
        self.eps_max          = eps_max
        self.eps_min          = eps_min
        self.eps_decay        = eps_decay
        self.n_epochs         = n_epochs
        self.batch_size       = batch_size
        self.buffer_size      = buffer_size

        self.experience_replay = []



        def q_network(state):
            input_state = InputLayer(input_var = state,
                                     shape     = (None, 
                                                  self.state_dimension[0], 
                                                  self.state_dimension[1], 
                                                  self.state_dimension[2]))

            input_state = DimshuffleLayer(input_state,
                                          pattern = (0,3,1,2))

            conv        = Conv2DLayer(input_state,
                                      num_filters  = 32,
                                      filter_size  = (8,8),
                                      stride       = (4,4),
                                      nonlinearity = rectify)

            conv        = Conv2DLayer(conv,
                                      num_filters  = 64,
                                      filter_size  = (4,4),
                                      stride       = (2,2),
                                      nonlinearity = rectify)

            conv        = Conv2DLayer(conv,
                                      num_filters  = 64,
                                      filter_size  = (3,3),
                                      stride       = (1,1),
                                      nonlinearity = rectify)
                  
            flatten     = FlattenLayer(conv)

            dense       = DenseLayer(flatten,
                                     num_units    = 512,
                                     nonlinearity = rectify)

            q_values    = DenseLayer(dense,
                                     num_units    = self.action_dimension,
                                     nonlinearity = linear)

            return q_values


        self.X_state          = T.ftensor4()
        self.X_action         = T.bvector()
        self.X_reward         = T.fvector()
        self.X_next_state     = T.ftensor4()
        self.X_done           = T.bvector()

        self.X_action_hot = to_one_hot(self.X_action, self.action_dimension)

        self.q_        = q_network(self.X_state);      self.q        = get_output(self.q_)
        self.q_target_ = q_network(self.X_next_state); self.q_target = get_output(self.q_target_)
        self.q_max     = T.max(self.q_target, axis=1)
        self.action    = T.argmax(self.q, axis=1)

        self.mu = theano.function(inputs               = [self.X_state],
                                  outputs              = self.action,
                                  allow_input_downcast = True)

        self.loss = squared_error(self.X_reward + self.gamma * self.q_max * (1.0 - self.X_done), 
                                  T.batched_dot(self.q, self.X_action_hot))
        self.loss = self.loss.mean()

        self.params = get_all_params(self.q_)

        self.updates = adam(self.loss,
                            self.params,
                            learning_rate = self.learning_rate)

        self.update_network = theano.function(inputs               = [self.X_state,
                                                                      self.X_action,
                                                                      self.X_reward,
                                                                      self.X_next_state,
                                                                      self.X_done],
                                              outputs              = self.loss,
                                              updates              = self.updates,
                                              allow_input_downcast = True)



    def preprocess(self,
                   state):
        return state[1:176:2, ::2, :]



    def get_action(self,
                   state, 
                   step):
        state = self.preprocess(state)

        eps = max(self.eps_min, 
                  self.eps_max - (self.eps_max - self.eps_min) * step / self.eps_decay)
        if np.random.rand() < eps:
            return np.random.randint(self.action_dimension)
        else:
            return self.mu([state])[0]




    def update_target(self):
        theta        = get_all_param_values(self.q_)
        set_all_param_values(self.q_target_, theta)



    def store_transition(self,
                         state,
                         action,
                         reward,
                         next_state,
                         done):

        state      = self.preprocess(state)
        next_state = self.preprocess(next_state)

        self.experience_replay.append((state, 
                                       action, 
                                       reward, 
                                       next_state, 
                                       done))
        self.experience_replay = self.experience_replay[-self.buffer_size:]



    def sample_transitions(self):
        return zip(*shuffle(self.experience_replay)[-self.batch_size:])



    def train(self):
        t_state, t_action, t_reward, t_next_state, t_done = self.sample_transitions()
        self.update_network(t_state,
                            t_action,
                            t_reward,
                            t_next_state,
                            t_done)



    def run(self):
        for epoch in range(self.n_epochs):
            state = self.env.reset()
            done  = False
            R     = 0.0
            step  = 0

            while not done:
                action = self.get_action(state,epoch)
                next_state, reward, done, info = self.env.step(action)
                self.store_transition(state, action, reward, next_state, done)
                
                if step % self.train_step == 0:
                    self.train()
                
                state = next_state

                R    += reward
                step += 1

            if epoch % self.critic_update == 0:
                self.update_target()

            print 'epoch: %d, reward: %f' % (epoch, R)



dqn = DeepQNetwork(atari_env        = 'SpaceInvaders-v4',
                   state_dimension  = np.array([88,80,3]),
                   action_dimension = 6,
                   train_step       = 4)

dqn.train()
