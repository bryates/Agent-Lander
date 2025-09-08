'''Replay Memory for storing and sampling experiences in Reinforcement Learning'''

import torch
import numpy as np

class ReplayMemory():
    '''Fixed-size buffer to store experience tuples.'''
    def __init__(self, input_dims, mem_size, batch_size, device='cpu'):
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mem_ptr = 0
        self.curr_state_memory = torch.zeros((self.mem_size, input_dims), device=device, dtype=torch.float32)
        self.next_state_memory = torch.zeros((self.mem_size, input_dims), device=device, dtype=torch.float32)
        self.action_memory = torch.zeros(self.mem_size, device=device, dtype=torch.int64)
        self.reward_memory = torch.zeros(self.mem_size, device=device, dtype=torch.float32)
        self.terminate_memory = torch.zeros(self.mem_size, device=device, dtype=torch.float32)

    def store(self, curr_state, next_state, action, reward, terminate):
        '''Store a new experience in the replay memory.'''
        mem_ptr = self.mem_ptr % self.mem_size  # Circular buffer
        self.curr_state_memory[mem_ptr] = curr_state
        self.next_state_memory[mem_ptr] = next_state
        self.action_memory[mem_ptr] = action
        self.reward_memory[mem_ptr] = reward
        self.terminate_memory[mem_ptr] = terminate

        self.mem_ptr += 1

    def sample(self):
        '''Randomly sample a batch of experiences from memory.'''
        mem_size = min(self.mem_ptr, self.mem_size)
        batch_index = np.random.choice(mem_size, self.batch_size, replace = False)

        return (
            self.curr_state_memory[batch_index],
            self.next_state_memory[batch_index],
            self.action_memory[batch_index],
            self.reward_memory[batch_index],
            self.terminate_memory[batch_index]
        )

    def ready(self):
        '''Check if there are enough samples in memory to sample a batch.'''
        return self.mem_ptr > self.batch_size
