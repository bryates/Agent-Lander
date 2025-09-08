'''Deep Q-Network (DQN) implementation for reinforcement learning tasks using PyTorch.'''
import random
import torch
import torch.nn as nn
import torch.optim as optim
import memory


# Define the Q-network
class QNetwork(nn.Module):
    '''Basic feedforward neural network for approximating Q-values.'''
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        '''Forward pass through the network.'''
        return self.model(x)


# Define the DQN agent
class DQNAgent:
    '''Deep Q-Network Agent for reinforcement learning tasks.'''
    def __init__(self, state_size, action_size, batch_size, hidden_size, lr=1e-3,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, device='cpu'):
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device

        self.q_network = QNetwork(state_size, action_size, hidden_size).to(device)
        self.q_target = QNetwork(state_size, action_size, hidden_size).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = memory.ReplayMemory(self.state_size, 10_000, self.batch_size, device=self.device)

    def act(self, state):
        '''Epsilon-greedy action selection.'''
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = state.to(device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        '''Store experience in replay memory.'''
        self.memory.store(state, next_state, action, reward, done)

    def replay(self):
        '''Sample a batch of experiences and perform a learning step.'''
        if not self.memory.ready():
            return
        states, next_states, actions, rewards, dones = self.memory.sample()
        states      = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones       = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.q_target(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

    def update_epsilon(self):
        '''Decay epsilon after each episode.'''
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def state_dict(self):
        '''Get the state dictionary of the Q-network.'''
        return self.q_network.state_dict()

    def load_state_dict(self, state_dict):
        '''Load the state dictionary into the Q-network.'''
        return self.q_network.load_state_dict(state_dict)

    def update_target(self):
        '''Update the target network to match the Q-network.'''
        self.q_target.load_state_dict(self.q_network.state_dict())
