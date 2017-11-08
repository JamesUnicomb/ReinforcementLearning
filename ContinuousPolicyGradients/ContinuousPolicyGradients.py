import os, inspect

import numpy as np
from math import *

import theano
import theano.tensor as T
from theano.tensor.extra_ops import to_one_hot, repeat
from theano.tensor.raw_random import multinomial

import lasagne
from lasagne.updates import adam
from lasagne.objectives import categorical_crossentropy
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, ElemwiseMergeLayer, get_output,  \
                           get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify, softmax, tanh, linear
from lasagne.init import Constant, Normal


class ContinuousPolicyGradients:
    def __init__(self,
                 n_state,
                 n_action,
                 scale_action        = 1.0,
                 mean_learning_rate  = 0.01,
                 sigma_learning_rate = 0.001,
                 gamma               = 0.99):

        self.n_state             = n_state
        self.n_action            = n_action
        self.scale_action        = scale_action
        self.mean_learning_rate  = mean_learning_rate
        self.sigma_learning_rate = sigma_learning_rate
        self.gamma               = gamma

        self.episode_state_history  = []
        self.episode_action_history = []
        self.episode_reward_history = []

        self.all_states  = []
        self.all_actions = []
        self.all_rewards = []

        def action_nonlinearity(x):
            return self.scale_action * tanh(x)

        # Neural Network for the policy

        def policy_network(state):
            input_state = InputLayer(input_var = state,
                                     shape     = (None, n_state))

            dense       = DenseLayer(input_state,
                                     num_units    = n_state,
                                     nonlinearity = tanh,
                                     W            = Normal(0.1, 0.0),
                                     b            = Constant(0.0))

            dense       = DenseLayer(dense,
                                     num_units    = n_state,
                                     nonlinearity = tanh,
                                     W            = Normal(0.1, 0.0),
                                     b            = Constant(0.0))

            mean        = DenseLayer(dense,
                                     num_units    = n_action,
                                     nonlinearity = action_nonlinearity,
                                     W            = Normal(0.1, 0.0),
                                     b            = Constant(0.0))

            sigma       = DenseLayer(dense,
                                     num_units    = n_action,
                                     nonlinearity = T.exp,
                                     W            = Normal(0.1, 0.0),
                                     b            = Constant(0.0))

            return mean, sigma


        # Defining the system variables (state, action, reward)

        self.X_state  = T.fmatrix()
        self.X_action = T.fmatrix()
        self.X_reward = T.fmatrix()

        # Policy and distribution functions

        self.policy_mean_, self.policy_sigma_ = policy_network(self.X_state)
        self.policy_mean                      = get_output(self.policy_mean_)
        self.policy_sigma                     = get_output(self.policy_sigma_)

        self.action_dist = theano.function(inputs               = [self.X_state],
                                           outputs              = [self.policy_mean, self.policy_sigma],
                                           allow_input_downcast = True)

        # log policy grads

        # d_f / d_u     = (action - mu) / sigma ^2
        # d_f / d_sigma = - 1 / sigma + (action - mu) ^ 2 / sigma ^3

        # E[d_J / d_u]     = (d_f / d_u) * R
        # E[d_J / d_sigma] =  (d_f / d_sigma) * R

        self.policy =  ( - 2 * T.log(self.policy_sigma) + (self.X_action - self.policy_mean) ** 2 * self.policy_sigma ** -2) * repeat(self.X_reward, n_action, axis = 1)
        self.policy = self.policy.mean()

        # Parameters to optimize

        self.mean_params  = get_all_params(self.policy_mean_)
        self.sigma_params = get_all_params(self.policy_sigma_)


        # Gradients w.r.t. Parameters

        self.mean_grads  = T.grad(self.policy, self.mean_params)
        self.sigma_grads = T.grad(self.policy, self.sigma_params)


        # Update equations

        self.mean_updates = adam(self.mean_grads,
                                 self.mean_params,
                                 learning_rate = self.mean_learning_rate)

        self.sigma_updates = adam(self.sigma_grads,
                                  self.sigma_params,
                                  learning_rate = self.sigma_learning_rate)

        self.update_mean_network = theano.function(inputs = [self.X_state,
                                                             self.X_action,
                                                             self.X_reward],
                                                   outputs = None,
                                                   updates = self.mean_updates,
                                                   allow_input_downcast = True)

        self.update_sigma_network = theano.function(inputs = [self.X_state,
                                                              self.X_action,
                                                              self.X_reward],
                                                    outputs = None,
                                                    updates = self.sigma_updates,
                                                    allow_input_downcast = True)

    def save_step_history(self,
                          state,
                          action,
                          reward):
        self.episode_state_history.append(state)
        self.episode_action_history.append(action)
        self.episode_reward_history.append(reward)


    def save_episode_history(self):
        self.all_states.append(self.episode_state_history)
        self.all_actions.append(self.episode_action_history)
        self.all_rewards.append(self.episode_reward_history)

        self.episode_state_history  = []
        self.episode_action_history = []
        self.episode_reward_history = []


    def update_network(self):
        rewards = self.discount_and_normalize_rewards(self.all_rewards, self.gamma)

        states  = np.array([state for episode_states in self.all_states for state in episode_states]).reshape(-1,self.n_state)
        actions = np.array([action for episode_actions in self.all_actions for action in episode_actions]).reshape(-1,self.n_action)
        rewards = np.array([reward for episode_rewards in rewards for reward in episode_rewards]).reshape(-1,1)

        self.update_mean_network(states,actions,rewards)
        self.update_sigma_network(states,actions,rewards)

        self.all_states  = []
        self.all_actions = []
        self.all_rewards = []


    def get_action(self,
                   state):
        mean, sigma = self.action_dist([state])
        mean = mean[0]
        sigma = sigma[0]

        return np.random.multivariate_normal(mean,np.diag(sigma ** 2))


    def discount_rewards(self,
                         rewards,
                         discount_rate):
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards


    def discount_and_normalize_rewards(self,
                                       all_rewards,
                                       discount_rate):
        all_discounted_rewards = [self.discount_rewards(rewards, discount_rate) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
