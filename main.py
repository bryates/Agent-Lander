'''Deep Q-Network (DQN) implementation for the
LunarLander-v3 environment using PyTorch and Gymnasium.'''

import os
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import model

# Hyperparameters
EPISODES = 2000  # Total number of training episodes
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

# Directory where videos will be saved
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# np.random.seed(SEED)
# torch.manual_seed(SEED)

# env = gym.make(ENV_NAME)
# env = gym.make(ENV_NAME, render_mode="human")
env = gym.make(ENV_NAME, render_mode="rgb_array")
env = RecordVideo(env, video_folder=VIDEO_DIR, episode_trigger=lambda ep: ep % 100 == 0)
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

def landing_penalty(curr_reward, curr_state, curr_action):
    '''Custom reward to reduce engine firing.'''
    # state = [x, y, x_vel, y_vel, angle, angular_vel, leg1_contact, leg2_contact]

    # On the ground and low velocity?
    landed = curr_state[6] == 1 and curr_state[7] == 1  # both legs on ground
    low_velocity = abs(curr_state[2]) < 0.1 and abs(curr_state[3]) < 0.1
    if landed and low_velocity and curr_action == 0:  # main engine
        curr_reward -= 0.5

    # Hovering with low velocity?
    low_vertical_vel = abs(curr_state[3]) < 0.05
    low_horizontal_vel = abs(curr_state[2]) < 0.05
    if not landed and low_vertical_vel and low_horizontal_vel:
        if curr_action == 0:  # main engine
            curr_reward -= 0.2
        elif curr_action in [1, 2]:  # side engines
            curr_reward -= 0.1

    return curr_reward

for episode in range(EPISODES):
    state, info = env.reset()
    # state, info = env.reset(seed=SEED)
    # print(f"Episode {episode+1}/{EPISODES}, Previous total reward: {total_reward}")
    if RENDER and total_reward > 100:
        env = gym.make(ENV_NAME, render_mode="human")
        env.reset()
        env.render()
    done = False
    total_reward = 0

    for _ in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        reward = landing_penalty(reward, state, action)
        state = (state - np.mean(state, axis=0)) / (np.std(state, axis=0) + 1e-6)
        next_state = (next_state - np.mean(next_state, axis=0)) / (np.std(next_state, axis=0) + 1e-6)
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
        print(f"Episode {episode}:\t\tAvg Reward (last {EVAL_INTERVAL}): {avg_reward:.2f}, ",
              f"Epsilon: {agent.epsilon:.3f}, Total Reward: {total_reward:.2f}")

    agent.update_epsilon()

moving_avg = np.convolve(rewards, np.ones((EVAL_INTERVAL,))/EVAL_INTERVAL, mode='valid')

plt.plot(rewards)
plt.plot(range(EVAL_INTERVAL-1, len(rewards)), moving_avg, color='red',
         label=f'{EVAL_INTERVAL}-episode Moving Average')  # smoothed
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
