'''Unit tests for the DQNAgent model.'''

import pytest
import numpy as np
import torch
import model


SEED = 42  # Random seed for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)               # PyTorch GPU
torch.cuda.manual_seed_all(SEED)           # If using multi-GPU
torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
torch.backends.cudnn.benchmark = False     # Disable benchmark for reproducibility

LEARNING_RATE = 1e-3  # Learning rate for the optimizer
STATE_SIZE = 8  # Size of the state space
ACTION_SIZE = 4  # Number of possible actions
HIDDEN_SIZE = 64  # Number of neurons in hidden layers
BATCH_SIZE = 256  # Minibatch size for experience replay
NUM_RANDOM_ACTIONS = 50  # Number of actions to sample when testing exploration
NUM_DECAY_STEPS = 1000  # Number of steps to test epsilon decay
GAMMA = 0.99  # Discount factor for future rewards
EPSILON_START = 0  # Disable for reproducibility
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.995  # Decay rate for exploration probability


@pytest.fixture(scope='module', autouse=True)
def agent():
    '''Fixture to create a fresh agent for each test.'''
    return model.DQNAgent(state_size=STATE_SIZE,
                          action_size=ACTION_SIZE,
                          batch_size=BATCH_SIZE,
                          hidden_size=HIDDEN_SIZE,
                          lr=LEARNING_RATE,
                          gamma=GAMMA,
                          epsilon_start=EPSILON_START,
                          epsilon_end=EPSILON_END,
                          epsilon_decay=EPSILON_DECAY)


def test_create_model(agent):
    '''Model should output the right shape given input state.'''
    x = torch.randn(1, STATE_SIZE)
    y = agent.q_network(x)
    assert y.shape == (1, ACTION_SIZE)


def test_remember(agent):
    '''Test remembering a transition.'''
    # Fill memory with at least BATCH_SIZE transitions
    for _ in range(BATCH_SIZE*2):
        state = torch.rand(STATE_SIZE, dtype=torch.float32)
        action = np.random.randint(ACTION_SIZE)
        reward = torch.rand(1)
        next_state = torch.rand(STATE_SIZE, dtype=torch.float32)
        done = torch.tensor(False, dtype=torch.float32)
        agent.remember(state, action, reward, next_state, done)


def test_act_deterministic(agent):
    '''With epsilon=0, actions should always be greedy and repeatable.'''
    agent.epsilon = 0.0
    state = torch.ones(STATE_SIZE)
    action1 = agent.act(state)
    action2 = agent.act(state)
    assert isinstance(action1, int)
    assert action1 == action2


def test_act_exploratory(agent):
    '''With epsilon=1, actions should come from the random branch.'''
    agent.epsilon = 1.0
    state = torch.ones(STATE_SIZE)
    actions = [agent.act(state) for _ in range(NUM_RANDOM_ACTIONS)]
    assert all(0 <= a < ACTION_SIZE for a in actions)
    assert len(set(actions)) > 1  # should not be the same every time


def test_replay(agent):
    '''Test training the model with experience replay.'''
    agent.replay()
    assert True


def test_learn_updates_weights(agent):
    '''A learning step should update model weights.'''
    params_before = [p.clone() for p in agent.q_network.parameters()]

    agent.replay()

    params_after = list(agent.q_network.parameters())
    changed = any((p1 - p2).abs().sum() > 0 for p1, p2 in zip(params_before, params_after))
    print(changed)
    assert changed


def test_epsilon_decay(agent):
    '''Epsilon should decay but not go below epsilon_end.'''
    start_epsilon = agent.epsilon
    agent.update_epsilon()
    assert agent.epsilon < start_epsilon
    for _ in range(NUM_DECAY_STEPS):
        agent.update_epsilon()
    assert np.round(agent.epsilon, 2) >= EPSILON_END


def test_save_model(agent):
    '''Test saving a DQNAgent model.'''
    if not agent:
        assert False, 'Could not find model!'
    torch.save(agent.state_dict(), 'agent.pth')
    assert True


def test_load_model(agent):
    ''' Test loading a DQNAgent model.'''
    if not agent:
        assert False, 'Could not find model!'
    agent_new = model.DQNAgent(state_size=STATE_SIZE,
                               action_size=ACTION_SIZE,
                               batch_size=BATCH_SIZE,
                               hidden_size=HIDDEN_SIZE,
                               lr=LEARNING_RATE,
                               gamma=GAMMA,
                               epsilon_start=EPSILON_START,
                               epsilon_end=EPSILON_END,
                               epsilon_decay=EPSILON_DECAY)
    agent_new.load_state_dict(torch.load('agent.pth'))
    for key in agent.state_dict():
        assert torch.all(agent.state_dict()[key] == agent_new.state_dict()[key])
    assert True
