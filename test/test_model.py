'''Unit tests for the DQNAgent model.'''

import pytest
import torch
import model


def pytest_namespace():
    '''Create namespace for sharing data between tests.'''
    return {'agent': None}


def test_create_model():
    '''Test creating a DQNAgent model.'''
    LEARNING_RATE = 1e-3  # Learning rate for the optimizer
    GAMMA = 0.99  # Discount factor for future rewards
    EPSILON_START = 1.0  # Initial exploration rate
    EPSILON_END = 0.01  # Final exploration rate
    EPSILON_DECAY = 0.995  # Decay rate for exploration probability

    agent = model.DQNAgent(state_size=8, action_size=4, hidden_size=64, lr=LEARNING_RATE,
                           gamma=GAMMA, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY)
    # request.config.cache.set('moddl', agent)
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
    assert True
