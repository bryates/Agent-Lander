'''Deep Q-Network (DQN) implementation for the
LunarLander-v3 environment using PyTorch and Gymnasium.'''

import warnings
import gymnasium as gym
warnings.filterwarnings(
    "ignore",
    message="builtin type .* has no __module__ attribute",
    category=DeprecationWarning,
)

def test_env():
    ENV_NAME = 'LunarLander-v3'  # Name of the Gym environment
    env = gym.make(ENV_NAME)

    '''
    EPISODES = 2

    for episode in range(EPISODES):
        landed_in_episode = False
        state, info = env.reset()

        print(state, info)
    '''

    assert True
