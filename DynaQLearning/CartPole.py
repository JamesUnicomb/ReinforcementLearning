# This algorithm performs Dyna-Q
# The Q-Learning is standard and for forward simulation a linear model is used.
#
# Algorithm Dyna-Q
# initialize replay memory with capacity buffer_size
# initialize models
#   q_action_network with parameters theta
#   q_target_network with parameters theta*
#   linead model f (such that f(state, action) = next_state, reward, done)
#
# for episode=1 to max_epochs do:
#   fit linear model to entire experience_replay set
#     f(state,action) = next_state, reward, done
#   for t=1 to Terminal do:
#     eps_greedy(state) = action
#     execute action, observe: reward, next_state, done
#     store (state, action, reward, next_state, done) in experience_replay
#     sample batch from experience replay (s, a, r, S, d)
#     bellman_equation:
#       loss = (r + gamma * argmax_A {q_target(S,A)} * (1 - d) - q_action(s,a))**2
#       update theta with respect to loss using backpropogation
#
#     update target network:
#       theta* = tau * theta + (1 - tau) * theta*
#
#     for k=1 to max_forward_sim do:
#       sample s, a from experience_replay
#       use linear model to predict S, r, d (with injected error)
#         bellman_equation:
#           loss = (r + gamma * argmax_a {q_target(S,a)} * (1 - d) - q_action(s,a))**2
#           update theta with respect to loss using backpropogation
#
#         update target network:
#           theta* = tau * theta + (1 - tau) * theta*
#
#       update q_action and q_target as above
#   end for
# end for

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
from lasagne.updates import adam, norm_constraint, total_norm_constraint
from lasagne.objectives import squared_error
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, BatchNormLayer, batch_norm, \
                           get_output, get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify, softmax, tanh
from lasagne.init import Constant, Normal

from LinearModel import LinearModel

n_state       = 4
n_action      = 2
learning_rate = 0.01
learning_tau  = 0.01
gamma         = 0.99
simulate_N    = 5

eps_max   = 0.5
eps_min   = 0.0
eps_decay = 200

n_epochs    = 200
batch_size  = 128
buffer_size = 40000

lr = LinearModel(n_state  = n_state,
                 n_action = n_action)

def q_network(state):
    input_state = InputLayer(input_var = state,
                             shape     = (None, n_state))

    dense       = DenseLayer(input_state,
                             num_units    = n_state,
                             nonlinearity = tanh)

    dense       = DenseLayer(dense,
                             num_units    = n_state,
                             nonlinearity = tanh)

    dense       = DenseLayer(dense,
                             num_units    = n_state,
                             nonlinearity = tanh)

    q_values    = DenseLayer(dense,
                             num_units    = n_action,
                             nonlinearity = None)

    return q_values

X_next_state     = T.fmatrix()
X_state          = T.fmatrix()
X_action         = T.bvector()
X_reward         = T.fvector()
X_done           = T.bvector()

X_action_hot = to_one_hot(X_action, n_action)

q_        = q_network(X_state);      q        = get_output(q_)
q_target_ = q_network(X_next_state); q_target = get_output(q_target_)
q_max     = T.max(q_target, axis=1)
action    = T.argmax(q, axis=1)

mu = theano.function(inputs               = [X_state],
                     outputs              = action,
                     allow_input_downcast = True)

loss = squared_error(X_reward + gamma * q_max * (1.0 - X_done), T.batched_dot(q, X_action_hot))
loss = loss.mean()

params = get_all_params(q_)

grads        = T.grad(loss,
                      params)

normed_grads = total_norm_constraint(grads, 1.0)

updates = adam(normed_grads,
               params,
               learning_rate = learning_rate)

update_network = theano.function(inputs               = [X_state,
                                                         X_action,
                                                         X_reward,
                                                         X_next_state,
                                                         X_done],
                                 outputs              = loss,
                                 updates              = updates,
                                 allow_input_downcast = True)

def get_action(state, step):
    eps = min(max(eps_min, eps_max - (eps_max - eps_min) * epoch / eps_decay), 1.0)
    if np.random.rand() < eps:
        return np.random.randint(n_action)
    else:
        return mu([state])[0]

def update_target():
    updates = []
    theta        = get_all_param_values(q_)
    theta_target = get_all_param_values(q_target_)
    for p, p_target in zip(*(theta, theta_target)):
        updates.append(learning_tau * p + (1 - learning_tau) * p_target)
    set_all_param_values(q_target_, updates)

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/cartpole_monitor', force=True)

experience_replay = []
all_steps = []
mean_steps = []

total_rewards = []
mean_rewards  = []

for epoch in range(n_epochs):
    state = env.reset()
    done  = False
    steps = 0

    rewards = []

    while not done:
        action = get_action(state,epoch)
        next_state, reward, done, info = env.step(action)

        experience_replay.append([state, action, reward, next_state, done])
        experience_replay = experience_replay[-buffer_size:]

        t_state, t_action, t_reward, t_next_state, t_done = zip(*shuffle(experience_replay)[-batch_size:])
        update_network(t_state, t_action, t_reward, t_next_state, t_done)
        update_target()

        if not epoch==0:
            for n in range(simulate_N):
                s_state, s_action, _, _, _ = zip(*shuffle(experience_replay)[-batch_size:])
                s_next_state, s_reward, s_done = lr.generate_sample(s_state, s_action)
                update_network(s_state, s_action, s_reward, s_next_state, s_done)
                update_target()

        rewards.append(reward)

        state = next_state

        steps += 1

    all_steps.append(steps)
    mean_step = np.mean(all_steps[-100:])
    mean_steps.append(mean_step)

    total_rewards.append(np.sum(rewards))
    mean_reward = np.mean(total_rewards[-100:])
    mean_rewards.append(mean_reward)

    m_state, m_action, m_reward, m_next_state, m_done = zip(*shuffle(experience_replay))
    lr.fit_models(m_state, m_action, m_reward, m_next_state, m_done)

    logging_info = 'epoch: %d, steps: %f, avg reward: %f, eps: %f' % (epoch, mean_step, mean_reward, min(max(eps_min, eps_max - (eps_max - eps_min) * epoch / eps_decay), 1.0))
    print logging_info


import matplotlib.pyplot as plt
plt.subplot(121)
plt.scatter(np.arange(len(all_steps)), all_steps, alpha=0.1, color='b')
plt.plot(mean_steps, color='k')
plt.title('Steps per Episode')

plt.subplot(122)
plt.scatter(np.arange(len(total_rewards)), total_rewards, alpha=0.1, color='r')
plt.plot(mean_rewards, color='k')
plt.plot([0,n_epochs], [195,195], 'k--')
plt.title('Cumulative Rewards')

plt.show()

import scipy.io as io
filename = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/results/data/cartpole'

io.savemat(filename, {'all_steps': all_steps,
                      'mean_steps': mean_steps,
                      'total_rewards': total_rewards,
                      'mean_rewards': mean_rewards})
