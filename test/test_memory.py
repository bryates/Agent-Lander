'''Unit tests for ReplayMemory class in memory.py'''

import torch
import pytest
from memory import ReplayMemory

# Constants for test configuration
STATE_SIZE = 8
MEM_SIZE = 10
BATCH_SIZE = 4
ACTION_SIZE = 4
OVERWRITE_MEM_SIZE = 5
OVERWRITE_BATCH_SIZE = 2

@pytest.fixture(scope='module', autouse=True)
def mem():
    return ReplayMemory(input_dims=STATE_SIZE, mem_size=MEM_SIZE, batch_size=BATCH_SIZE)

def test_ready(mem):
    '''Test the ready method for checking if enough samples are available.'''
    assert not mem.ready()
    for _ in range(BATCH_SIZE):
        mem.store(torch.zeros(STATE_SIZE), torch.zeros(STATE_SIZE), 0, 0, False)
    assert not mem.ready()
    mem.store(torch.zeros(STATE_SIZE), torch.zeros(STATE_SIZE), 0, 0, False)
    assert mem.ready()

def test_store_and_sample(mem):
    '''Test storing and sampling experiences.'''
    state = torch.arange(STATE_SIZE, dtype=torch.float32)
    next_state = torch.arange(STATE_SIZE, dtype=torch.float32) + 1
    for i in range(MEM_SIZE):
        mem.store(state + i, next_state + i, i, i % ACTION_SIZE, i % 2 == 0)
    batch = mem.sample()
    assert batch[0].shape == (BATCH_SIZE, STATE_SIZE)
    assert batch[1].shape == (BATCH_SIZE, STATE_SIZE)
    assert batch[2].shape == (BATCH_SIZE,)
    assert batch[3].shape == (BATCH_SIZE,)

def test_circular_buffer():
    '''Test that old experiences are overwritten when memory is full.'''
    mem = ReplayMemory(input_dims=STATE_SIZE, mem_size=OVERWRITE_MEM_SIZE, batch_size=OVERWRITE_BATCH_SIZE)
    for i in range(OVERWRITE_MEM_SIZE + 2):
        mem.store(torch.ones(STATE_SIZE) * i, torch.ones(STATE_SIZE) * (i+1), i, i, False)
    assert mem.mem_ptr == OVERWRITE_MEM_SIZE + 2
    # The oldest experiences should be overwritten
    expected_oldest = torch.ones(STATE_SIZE) * (OVERWRITE_MEM_SIZE)
    expected_second = torch.ones(STATE_SIZE) * (OVERWRITE_MEM_SIZE + 1)
    assert torch.all(mem.curr_state_memory[0] == expected_oldest)
    assert torch.all(mem.curr_state_memory[1] == expected_second)
