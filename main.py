'''Deep Q-Network (DQN) implementation for the
LunarLander-v3 environment using PyTorch and Gymnasium.'''

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import model

# Hyperparameters
EPISODES = 2000  # Total number of training episodes
BATCH_SIZE = 64  # Minibatch size for experience replay
TARGET_UPDATE = 10  # Update target network every TARGET_UPDATE episodes
MAX_STEPS = 200  # Max steps per episode
EVAL_EPISODES = 10  # Number of episodes for evaluation
EVAL_INTERVAL = 50  # Evaluate the agent every EVAL_INTERVAL episodes
LEARNING_RATE = 1e-3  # Learning rate for the optimizer
GAMMA = 0.99  # Discount factor for future rewards
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.995  # Decay rate for exploration probability
HIDDEN_SIZE = 64  # Number of neurons in hidden layers
ENV_NAME = 'LunarLander-v3'  # Name of the Gym environment
RENDER = False  # Whether to render the environment
SEED = 42  # Random seed for reproducibility

# np.random.seed(SEED)
# torch.manual_seed(SEED)

env = gym.make(ENV_NAME)
# env = gym.make(ENV_NAME, render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = model.DQNAgent(state_size, action_size,
                       hidden_size=HIDDEN_SIZE,
                       lr=LEARNING_RATE,
                       gamma=GAMMA,
                       epsilon_start=EPSILON_START,
                       epsilon_end=EPSILON_END,
                       epsilon_decay=EPSILON_DECAY)

total_reward = 0
rewards = []
for episode in range(EPISODES):
    state, info = env.reset()
    # state, info = env.reset(seed=SEED)
    # print(f"Episode {episode+1}/{EPISODES}, Previous total reward: {total_reward}")
    if RENDER and total_reward > 0:
        env.render(render_mode="human")
    done = False
    total_reward = 0

    for _ in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = (state - np.mean(state, axis=0)) / (np.std(state, axis=0) + 1e-6)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay(BATCH_SIZE)

        if done:
            break

    if episode % TARGET_UPDATE == 0:
        agent.update_target()

    rewards.append(total_reward)

    if episode % EVAL_INTERVAL == 0:
        avg_reward = np.mean(rewards[-50:])
        print(f"Episode {episode}, Avg Reward (last {EVAL_INTERVAL}): {avg_reward:.2f}, \
              Epsilon: {agent.epsilon:.3f}")

    agent.update_epsilon()

moving_avg = np.convolve(rewards, np.ones((EVAL_INTERVAL,))/EVAL_INTERVAL, mode='valid')

plt.plot(rewards)
plt.plot(range(EVAL_INTERVAL-1, len(rewards)), moving_avg, color='red',
         label=f'{EVAL_INTERVAL}-episode Moving Average')  # smoothed
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
