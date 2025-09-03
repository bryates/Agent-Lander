'''Unit tests for the DQNAgent model.'''

import pytest
import torch
import numpy as np
import os
import model


SEED = 42  # Random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)               # PyTorch GPU
torch.cuda.manual_seed_all(SEED)           # If using multi-GPU
torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
torch.backends.cudnn.benchmark = False     # Disable benchmark for reproducibility
BATCH_SIZE = 256  # Minibatch size for experience replay


def pytest_namespace():
    '''Create namespace for sharing data between tests.'''
    return {'agent': None}


def test_create_model():
    '''Test creating a DQNAgent model.'''
    LEARNING_RATE = 1e-3  # Learning rate for the optimizer
    GAMMA = 0.99  # Discount factor for future rewards
    EPSILON_START = 0  # Disable for reproducibility
    EPSILON_END = 0.01  # Final exploration rate
    EPSILON_DECAY = 0.995  # Decay rate for exploration probability

    agent = model.DQNAgent(state_size=8, action_size=4, hidden_size=64, lr=LEARNING_RATE,
                           gamma=GAMMA, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY)
    # request.config.cache.set('moddl', agent)
    pytest.agent = agent
    assert True


def test_remember():
    agent = pytest.agent

    state = np.random.rand(BATCH_SIZE)
    action = np.random.randint(0, 5, BATCH_SIZE)
    reward = np.random.randint(0, BATCH_SIZE, BATCH_SIZE)
    next_state = np.random.randint(0, 5, BATCH_SIZE)
    done = np.random.randint(0, 2, BATCH_SIZE)

    agent.remember(state, action, reward, next_state, done)

    pytest.agent = agent
    assert True


def test_predict():
    agent = pytest.agent
    state = np.random.rand(8)
    action = agent.act(state)
    assert action == 1


def test_replay():
    agent = pytest.agent
    agent.replay(BATCH_SIZE)
    pytest.agent = agent
    assert True


def test_save_model():
    '''Test saving a DQNAgent model.'''
    # agent = request.config.cache.get('moddl', None)
    agent = pytest.agent
    if not agent:
        assert False, 'Could not find model!'
    torch.save(agent.state_dict(), 'agent.pth')
    assert True


def test_load_model():
    ''' Test loading a DQNAgent model.'''
    LEARNING_RATE = 1e-3  # Learning rate for the optimizer
    GAMMA = 0.99  # Discount factor for future rewards
    EPSILON_START = 1.0  # Initial exploration rate
    EPSILON_END = 0.01  # Final exploration rate
    EPSILON_DECAY = 0.995  # Decay rate for exploration probability

    agent = pytest.agent
    if not agent:
        assert False, 'Could not find model!'
    agent_new = model.DQNAgent(state_size=8, action_size=4, hidden_size=64, lr=LEARNING_RATE,
                               gamma=GAMMA, epsilon_start=EPSILON_START,
                               epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY)
    agent_new.load_state_dict(torch.load('agent.pth'))
    for key in agent.state_dict():
        assert torch.all(agent.state_dict()[key] == agent_new.state_dict()[key])
    assert True
