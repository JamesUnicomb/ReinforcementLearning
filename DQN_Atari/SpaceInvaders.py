from DQN import DeepQNetwork

dqn = DeepQNetwork(atari_env        = 'SpaceInvaders-v4',
                   state_dimension  = [88,80,3],
                   action_dimension = 6,
                   train_step       = 4)

dqn.run()
