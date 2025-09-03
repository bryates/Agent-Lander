'''Deep Q-Network (DQN) implementation for the 
LunarLander-v3 environment using PyTorch and Gymnasium.'''

import gymnasium as gym
from matplotlib import pyplot as plt
import model

# Hyperparameters
EPISODES = 500 # Total number of training episodes
BATCH_SIZE = 64 # Minibatch size for experience replay
TARGET_UPDATE = 100 # Update target network every TARGET_UPDATE episodes
MAX_STEPS = 200 # Max steps per episode
EVAL_EPISODES = 10 # Number of episodes for evaluation
EVAL_INTERVAL = 50 # Evaluate the agent every EVAL_INTERVAL episodes
LEARNING_RATE = 1e-3 # Learning rate for the optimizer
GAMMA = 0.99 # Discount factor for future rewards
EPSILON_START = 1.0 # Initial exploration rate
EPSILON_END = 0.01 # Final exploration rate
EPSILON_DECAY = 0.995 # Decay rate for exploration probability
HIDDEN_SIZE = 64 # Number of neurons in hidden layers
ENV_NAME = 'LunarLander-v3' # Name of the Gym environment
RENDER = False # Whether to render the environment
SEED = 42 # Random seed for reproducibility

#np.random.seed(SEED)
#torch.manual_seed(SEED)

env = gym.make(ENV_NAME)#, render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = model.DQNAgent(state_size, action_size,
                        lr=LEARNING_RATE,
                        gamma=GAMMA,
                        epsilon_start=EPSILON_START,
                        epsilon_end=EPSILON_END,
                        epsilon_decay=EPSILON_DECAY)

total_reward = 0
rewards = []
for episode in range(EPISODES):
    state, info = env.reset()#seed=SEED)
    print(f"Episode {episode+1}/{EPISODES}, Previous total reward: {total_reward}")
    done = False
    total_reward = 0
    if RENDER:
        env.render()

    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay(BATCH_SIZE)

    if episode % TARGET_UPDATE == 0:
        agent.q_target.load_state_dict(agent.q_network.state_dict())

    rewards.append(total_reward)

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
