import numpy as np
from numpy.random import multivariate_normal
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class LinearModel:
    def __init__(self,
                 n_state,
                 n_action):

        self.lr_state  = LinearRegression()
        self.lr_reward = LinearRegression()
        self.lr_done   = LogisticRegression()

        self.state_covariance  = np.zeros((n_state, n_state))
        self.reward_covariance = np.zeros((1, 1))

        self.ohe = OneHotEncoder(sparse=False)

    def fit_models(self,
                   state,
                   action,
                   reward,
                   next_state,
                   done):
        action = self.ohe.fit_transform(np.array(action).reshape(-1,1))
        reward = np.array(reward).reshape(-1,1)

        X = np.concatenate([state, action], axis=1)

        self.lr_state.fit(X, next_state)
        self.lr_reward.fit(X, reward)
        self.lr_done.fit(X, done)

        state_error = np.matrix(self.lr_state.predict(X) - state)
        reward_error = np.matrix(self.lr_reward.predict(X) - reward)

        self.state_covariance  = state_error.T * state_error / (state_error.shape[0] - state_error.shape[1])
        self.reward_covariance = reward_error.T * reward_error / (reward_error.shape[0] - reward_error.shape[1])

    def generate_sample(self,
                        state,
                        action):
        action = self.ohe.fit_transform(np.array(action).reshape(-1,1))

        X = np.concatenate([state, action], axis=1)

        next_state = np.array([multivariate_normal(x, self.state_covariance) for x in self.lr_state.predict(X)])
        reward     = np.array([multivariate_normal(r, self.reward_covariance) for r in self.lr_reward.predict(X)])
        done       = np.array(self.lr_done.predict(X))

        return next_state, reward.reshape(-1), done.reshape(-1)
