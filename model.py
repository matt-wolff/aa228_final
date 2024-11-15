import torch
import torch.nn as nn
import numpy as np
import random

NUM_WILD_CARDS = 13
STATE_DIM = 135  # First thirteen are wild cards
ACTION_DIM = 11  # check, call, fold, raise (2x, 3x, 4x highest bet. 1/4, 1/2, 1, 2 pot. all-in)

class PokerModel(nn.Module):
    def __init__(self,
                 hidden_dim,
                 n_layers,
                 dropout_p,
                 vanilla  # If true, don't consider the first 
                 ):
        self.vanilla = vanilla

        layers = []
        prev_dim = STATE_DIM - NUM_WILD_CARDS if vanilla else  STATE_DIM
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, ACTION_DIM))
        self.model = nn.Sequential(layers)

        self.epsilion = 1
        self.epsilon_lr = 1 - 10**-4

    def next_action(self, state_vector):
        if self.vanilla:
            state_vector = state_vector[NUM_WILD_CARDS:]
        random_action = random.random() < self.epsilion
        if random_action:
            return random.randint(0, ACTION_DIM-1)
        logits = self.model(state_vector)
        self.epsilion = self.epsilion * self.epsilon_lr
        return np.argmax(logits)

