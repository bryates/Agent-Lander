'''Deep Q-Network (DQN) implementation for the
LunarLander-v3 environment using PyTorch and Gymnasium.'''

import os
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import model

# Hyperparameters
EPISODES = 3000  # Total number of training episodes
BATCH_SIZE = 64  # Minibatch size for experience replay
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
y_values = []
yv_values = []


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

    if not landed and low_vertical_vel:
        if curr_action == 0:  # main engine
            curr_reward -= 0.2

    low_horizontal_vel = abs(curr_state[2]) < 0.05

    if not landed and low_vertical_vel and low_horizontal_vel:
        if curr_action in [1, 2]:  # side engines
            curr_reward -= 0.1

    return curr_reward


def normalize_state(state):
    '''Normalize state features to range [-1, 1] where applicable.'''
    # State: [x, y, x_vel, y_vel, angle, angular_vel, leg1, leg2]
    norm_state = np.zeros_like(state, dtype=np.float32)
    norm_state[0] = state[0] / 1.5        # x
    norm_state[1] = state[1] / 1.0        # y
    norm_state[2] = state[2] / 5.0        # x_vel
    norm_state[3] = state[3] / 5.0        # y_vel
    norm_state[4] = state[4] / np.pi      # angle
    norm_state[5] = state[5] / 5.0        # angular_vel
    norm_state[6] = state[6]              # leg1 contact (0 or 1)
    norm_state[7] = state[7]              # leg2 contact (0 or 1)
    return norm_state


for episode in range(EPISODES):
    landed_in_episode = False
    state, info = env.reset()
    # state, info = env.reset(seed=SEED)
    # print(f"Episode {episode+1}/{EPISODES}, Previous total reward: {total_reward}")
    if RENDER and total_reward > 100:
        env = gym.make(ENV_NAME, render_mode="human")
        env.reset()
        env.render()
    done = False
    total_reward = 0

    initial_y = state[1]  # initial vertical position
    initial_yvel = state[3]  # initial vertical velocity

    for i in range(MAX_STEPS):
        # state = normalize_state(state)
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        # reward = landing_penalty(reward, state, action)
        # next_state = normalize_state(next_state)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay(BATCH_SIZE)

        if done:
            break
    final_y = state[1]
    final_yvel = state[3]
    y_values.append((initial_y, final_y))
    yv_values.append((initial_yvel, final_yvel * 5))

    if episode % TARGET_UPDATE == 0:
        agent.update_target()

    rewards.append(total_reward)

    if episode % EVAL_INTERVAL == 0:
        avg_reward = np.mean(rewards[-50:])
        print(f"Episode {episode}:\t\tAvg Reward (last {EVAL_INTERVAL}): {avg_reward:.2f}, ",
              f"Epsilon: {agent.epsilon:.3f}, Total Reward (last episode): {total_reward:.2f}")

    agent.update_epsilon()

moving_avg = np.convolve(rewards, np.ones((EVAL_INTERVAL,))/EVAL_INTERVAL, mode='valid')

plt.plot([y[0] for y in y_values], label='Initial Y')
plt.plot([y[1] for y in y_values], label='Final Y')
plt.xlabel('Episode')
plt.ylabel('Y Position')
plt.legend()
plt.savefig('y_positions.png')
plt.show()
plt.close()

plt.plot([yv[0] for yv in yv_values], label='Initial Y Velocity')
plt.plot([yv[1] for yv in yv_values], label='Final Y Velocity')
plt.xlabel('Episode')
plt.ylabel('Y Velocity')
plt.legend()
plt.savefig('y_velocities.png')
plt.show()
plt.close()

plt.plot(rewards)
plt.plot(range(EVAL_INTERVAL-1, len(rewards)), moving_avg, color='red',
         label=f'{EVAL_INTERVAL}-episode Moving Average')  # smoothed
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.savefig('training_rewards.png')
plt.show()
plt.close()
