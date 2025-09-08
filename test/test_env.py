'''Deep Q-Network (DQN) implementation for the
LunarLander-v3 environment using PyTorch and Gymnasium.'''

import warnings
import torch
import gymnasium as gym
import model
warnings.filterwarnings(
    "ignore",
    message="builtin type .* has no __module__ attribute",
    category=DeprecationWarning,
)

# Hyperparameters
EPISODES = 2  # Total number of training episodes
BATCH_SIZE = 256  # Minibatch size for experience replay
TARGET_UPDATE = 10  # Update target network every TARGET_UPDATE episodes
MAX_STEPS = 500  # Max steps per episode
EVAL_EPISODES = 10  # Number of episodes for evaluation
EVAL_INTERVAL = 50  # Evaluate the agent every EVAL_INTERVAL episodes
LEARNING_RATE = 5e-4  # Learning rate for the optimizer
GAMMA = 0.99  # Discount factor for future rewards
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.99  # Decay rate for exploration probability
HIDDEN_SIZE = 64  # Number of neurons in hidden layers
ENV_NAME = 'LunarLander-v3'  # Name of the Gym environment
RENDER = False  # Whether to render the environment
SEED = 42  # Random seed for reproducibility

torch.manual_seed(SEED)

def test_env():
    '''Test if the environment works with the agent.'''
    env = gym.make(ENV_NAME)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = model.DQNAgent(state_size, action_size,
                           batch_size=BATCH_SIZE,
                           hidden_size=HIDDEN_SIZE,
                           lr=LEARNING_RATE,
                           gamma=GAMMA,
                           epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END,
                           epsilon_decay=EPSILON_DECAY)

    total_reward = 0
    for _ in range(EPISODES):
        state, _ = env.reset(seed=SEED)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(0)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay()

    assert True
