import gym

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
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, get_output, \
                           get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify, softmax, tanh
from lasagne.init import Constant, Normal

n_state       = 4
n_action      = 2
learning_rate = 0.001
learning_tau  = 0.001
gamma         = 0.99

eps_max   = 1.00
eps_min   = 0.01
eps_decay = 0.9975

n_epochs    = 1000
batch_size  = 32
buffer_size = 40000

def q_network(state):
    input_state = InputLayer(input_var = state,
                             shape     = (None, n_state))

    dense_1     = DenseLayer(input_state,
                             num_units    = n_state,
                             nonlinearity = tanh,
                             W         = Normal(0.1, 0.0),
                             b         = Constant(0.0))

    dense_2     = DenseLayer(dense_1,
                             num_units    = n_state,
                             nonlinearity = tanh,
                             W         = Normal(0.1, 0.0),
                             b         = Constant(0.0))

    q_values    = DenseLayer(dense_2,
                             num_units    = n_action,
                             nonlinearity = None,
                             W         = Normal(0.1, 0.0),
                             b         = Constant(0.0))

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

updates = adam(loss,
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
    eps = max(eps_min, eps_min + (eps_max - eps_min) * eps_decay ** step)
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

experience_replay = []
all_steps = []
mean_steps = []

for epoch in range(n_epochs):
    state = env.reset()
    done  = False
    steps = 0

    while not done:
        action = get_action(state,epoch)
        next_state, reward, done, info = env.step(action)

        experience_replay.append([state, action, reward, next_state, done])
        experience_replay = experience_replay[-buffer_size:]

        t_state, t_action, t_reward, t_next_state, t_done = zip(*shuffle(experience_replay)[-batch_size:])
        update_network(t_state, t_action, t_reward, t_next_state, t_done)
        update_target()

        state = next_state

        steps += 1

    all_steps.append(steps)
    mean_step = np.mean(all_steps[-100:])
    mean_steps.append(mean_step)

    if epoch % 25 == 0:
        print 'epoch: %d, steps: %f, eps: %f' % (epoch, mean_step, max(eps_min, eps_min + (eps_max - eps_min) * eps_decay ** epoch))

import matplotlib.pyplot as plt
plt.scatter(np.arange(len(all_steps)), all_steps, alpha=0.1)
plt.plot(mean_steps)
plt.ylim([0,200])
plt.show()
