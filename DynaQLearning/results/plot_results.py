import os, inspect

import numpy as np
from math import *

import matplotlib.pyplot as plt

import scipy.io as io

filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/data/'
files    = os.listdir(filepath)

files = files * 3

#rows = ['Acrobot', 'CartPole', 'LunarLander']
rows = ['CartPole','CartPole','CartPole']

f, axarr = plt.subplots(len(files), 2, figsize = (12,10))

for i, datapath in enumerate(files):
    data = io.loadmat(filepath + datapath)

    all_steps     = data['all_steps'][0]
    mean_steps    = data['mean_steps'][0]
    total_rewards = data['total_rewards'][0]
    mean_rewards  = data['mean_rewards'][0]


    axarr[i,0].scatter(np.arange(len(all_steps)), all_steps, alpha=0.5, s=2.0, color=(0.7, 0.7, 0.7))
    axarr[i,0].plot(mean_steps, color='b', linewidth=3)
    axarr[i,0].set_ylabel(rows[i], fontsize=20)
    axarr[i,0].yaxis.set_label_coords(-0.15,0.3)
    if i==0:
        axarr[i,0].set_title('Number of steps per Episode', fontsize=20)

    axarr[i,1].scatter(np.arange(len(total_rewards)), total_rewards, alpha=0.5, s=2.0, color=(0.7, 0.7, 0.7))
    axarr[i,1].plot(mean_rewards, color='r', linewidth=3)
    if i==0:
        axarr[i,1].set_title('Cumulative Rewards per Episode', fontsize=20)

plt.show()
