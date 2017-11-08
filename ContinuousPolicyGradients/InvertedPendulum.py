import gym

import os, inspect

import numpy as np
from math import *

from ContinuousPolicyGradients import ContinuousPolicyGradients

monitor     = True

n_epochs    = 5000
n_state     = 4
n_action    = 1
update_step = 10
all_rewards = []

cpg = ContinuousPolicyGradients(n_state             = n_state,
                                n_action            = n_action,
                                scale_action        = 3.0,
                                mean_learning_rate  = 0.005,
                                sigma_learning_rate = 0.001,
                                gamma               = 0.995)

env = gym.make('InvertedPendulum-v1')
if monitor:
    env = gym.wrappers.Monitor(env, os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/invertedpendulum_monitor', force=True)


for epoch in range(n_epochs):
    state = env.reset()
    done  = False
    steps = 0
    R = 0

    while not done:
        action = cpg.get_action(state)
        next_state, reward, done, info = env.step(action)

        cpg.save_step_history(state, action, reward)

        state = next_state

        R += reward
        steps += 1

    cpg.save_episode_history()

    all_rewards.append(R)

    if epoch % update_step == 0:
        cpg.update_network()

    if epoch % 50 == 0:
        print 'epoch: %d, rewards: %f' % (epoch, np.mean(all_rewards[-100:]))
