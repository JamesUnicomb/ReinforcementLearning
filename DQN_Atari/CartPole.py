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


class QNetwork:
    def __init__(self,
                 gym_env,
                 state_dimension,
                 action_dimension,
                 monitor_env          = False,
                 print_episode_info   = 25,
                 learning_rate_actor  = 0.01,
                 learning_rate_critic = 0.01,
                 gamma                = 0.95,
                 eps_max              = 1.0,
                 eps_min              = 0.1,
                 eps_decay            = 1000,
                 n_epochs             = 1000,
                 batch_size           = 64,
                 buffer_size          = 200000):

        self.env = gym.make(gym_env)
        if monitor_env:
            None

        self.print_episode_info   = print_episode_info

        self.state_dimension      = state_dimension
        self.action_dimension     = action_dimension
        self.learning_rate_actor  = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma                = gamma
        self.eps_max              = eps_max
        self.eps_min              = eps_min
        self.eps_decay            = eps_decay
        self.n_epochs             = n_epochs
        self.batch_size           = batch_size
        self.buffer_size          = buffer_size

        self.experience_replay = []



        def q_network(state):
            input_state = InputLayer(input_var = state,
                                     shape     = (None, state_dimension))

            dense       = DenseLayer(input_state,
                                     num_units    = action_dimension,
                                     nonlinearity = tanh)

            dense       = DenseLayer(dense,
                                     num_units    = action_dimension,
                                     nonlinearity = tanh)

            q_values    = DenseLayer(dense,
                                     num_units    = action_dimension,
                                     nonlinearity = linear)

            return q_values


        self.X_state          = T.fmatrix()
        self.X_action         = T.bvector()
        self.X_reward         = T.fvector()
        self.X_next_state     = T.fmatrix()
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
                            learning_rate = self.learning_rate_actor)

        self.update_network = theano.function(inputs               = [self.X_state,
                                                                      self.X_action,
                                                                      self.X_reward,
                                                                      self.X_next_state,
                                                                      self.X_done],
                                              outputs              = self.loss,
                                              updates              = self.updates,
                                              allow_input_downcast = True)



    def get_action(self,
                   state,
                   step):
        eps = max(self.eps_min,
                  self.eps_max - (self.eps_max - self.eps_min) * step / self.eps_decay)

        if np.random.rand() < eps:
            return np.random.randint(self.action_dimension)
        else:
            return self.mu([state])[0]



    def update_target(self):
        updates = []
        theta        = get_all_param_values(self.q_)
        theta_target = get_all_param_values(self.q_target_)

        for p, p_target in zip(*(theta, theta_target)):
            updates.append(self.learning_rate_critic * p + (1 - self.learning_rate_critic) * p_target)

        set_all_param_values(self.q_target_, updates)



    def store_transition(self,
                         state,
                         action,
                         reward,
                         next_state,
                         done):

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
        all_rewards = []

        for epoch in range(self.n_epochs):
            state = self.env.reset()
            done  = False
            R     = 0.0

            while not done:
                action = self.get_action(state,epoch)
                next_state, reward, done, info = self.env.step(action)

                self.store_transition(state, action, reward, next_state, done)
                self.train()
                self.update_target()

                R += reward
                state = next_state

            all_rewards.append(R)

            if epoch % self.print_episode_info == 0:
                print 'epoch: %d, avg. reward: %f' % (epoch, np.mean(all_rewards[-100:]))


qn = QNetwork(gym_env          = 'CartPole-v0',
              state_dimension  = 4,
              action_dimension = 2)

qn.run()
