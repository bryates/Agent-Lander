'''Deep Q-Network (DQN) implementation for the
LunarLander-v3 environment using PyTorch and Gymnasium.'''

import gymnasium as gym
import Box2D
Box2D.b2.SWIG_PyPacked.__module__ = "Box2D"
Box2D.b2.SWIG_PyObject.__module__ = "Box2D"

ENV_NAME = 'LunarLander-v3'  # Name of the Gym environment
env = gym.make(ENV_NAME)

'''
EPISODES = 2

for episode in range(EPISODES):
    landed_in_episode = False
    state, info = env.reset()

    print(state, info)
'''
