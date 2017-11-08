import gym

import os, inspect

import numpy as np
from math import *

import theano
import theano.tensor as T
from theano.tensor.extra_ops import to_one_hot
from theano.tensor.raw_random import multinomial

import lasagne
from lasagne.updates import adam
from lasagne.objectives import categorical_crossentropy
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, get_output, \
                           get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify, softmax, tanh
from lasagne.init import Constant, Normal


n_input       = 8
n_output      = 4
learning_rate = 0.01

n_epochs = 4000
gamma = 0.99

update_step = 10

def policy_network(state):
    input_state = InputLayer(input_var = state,
                             shape     = (None, n_input))

    dense_1     = DenseLayer(input_state,
                             num_units    = n_input,
                             nonlinearity = tanh)

    dense_2     = DenseLayer(dense_1,
                             num_units    = n_input,
                             nonlinearity = tanh)

    probs       = DenseLayer(dense_2,
                             num_units    = n_output,
                             nonlinearity = softmax)

    return probs

X_state          = T.fmatrix()
X_action         = T.bvector()
X_reward         = T.fvector()

X_action_hot = to_one_hot(X_action, n_output)

prob_values = policy_network(X_state)

policy_ = get_output(prob_values)
policy  = theano.function(inputs               = [X_state],
                          outputs              = policy_,
                          allow_input_downcast = True)

loss = categorical_crossentropy(policy_, X_action_hot) * X_reward
loss = loss.mean()

params = get_all_params(prob_values)

updates = adam(loss,
               params,
               learning_rate = learning_rate)

update_network = theano.function(inputs               = [X_state,
                                                         X_action,
                                                         X_reward],
                                 outputs              = loss,
                                 updates              = updates,
                                 allow_input_downcast = True)

def get_action(state):
    probs = policy([state])[0]
    action = np.argmax(np.random.multinomial(1,probs))
    return action

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

env = gym.make('LunarLander-v2')
env = gym.wrappers.Monitor(env, os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/lunarlander_monitor', force=True)

all_states  = []
all_actions = []
all_rewards = []

all_steps  = []
mean_steps = []

total_rewards = []
mean_rewards  = []

for epoch in range(n_epochs):
    state = env.reset()
    states  = []
    actions = []
    rewards = []
    steps   = 0
    done    = False

    while not done:
        action = get_action(state)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

        steps += 1

    total_rewards.append(np.sum(rewards))
    mean_reward = np.mean(total_rewards[-100:])
    mean_rewards.append(mean_reward)

    mean_step = np.mean(all_steps[-100:])
    mean_steps.append(mean_step)

    all_states  += list(states)
    all_actions += list(actions)
    all_rewards.append(rewards)
    all_steps.append(steps)

    if epoch%update_step == 0:
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate = gamma)
        all_rewards = [item for sublist in all_rewards for item in sublist]

        update_network(all_states, all_actions, all_rewards)

        all_states  = []
        all_actions = []
        all_rewards = []

    if epoch%25 == 0:
        print 'epoch: %d, steps: %f, avg. reward: %f' % (epoch, mean_step, mean_reward)

import matplotlib.pyplot as plt
plt.subplot(121)
plt.scatter(np.arange(len(all_steps)), all_steps, alpha=0.1, color='b')
plt.plot(mean_steps, color='k')
plt.title('Steps per Episode')

plt.subplot(122)
plt.scatter(np.arange(len(total_rewards)), total_rewards, alpha=0.1, color='r')
plt.plot([0,n_epochs], [200,200], 'k--')
plt.plot(mean_rewards, color='k')
plt.title('Cumulative Rewards')

plt.show()

import scipy.io as io
filename = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/results/data/lunarlander'

io.savemat(filename, {'all_steps': all_steps,
                      'mean_steps': mean_steps,
                      'total_rewards': total_rewards,
                      'mean_rewards': mean_rewards})
