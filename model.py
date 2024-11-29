from torch.optim import AdamW
import torch.nn as nn
import numpy as np
import random
import torch

NUM_WILD_CARDS = 13
STATE_DIM = 136  # First thirteen are wild cards
ACTION_DIM = 11  # check, call, fold, raise (2x, 3x, 4x highest bet. 1/4, 1/2, 1, 2 pot. all-in)

class PokerModel(nn.Module):
    def __init__(
            self,
            hidden_dim,
            n_layers,
            dropout_p,
        ):
        super().__init__()

        layers = []
        prev_dim = STATE_DIM
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, ACTION_DIM))
        self.model = nn.Sequential(*layers)
        self.optimizer = AdamW(self.model.parameters())

        self.epsilion = 1
        self.epsilon_lr = 1 - 10**-4

    def max_action(self, state_vector):
        logits = self.model(state_vector)
        action = torch.argmax(logits)
        return action, logits[action]

    def next_action(self, state_vector, game):
        random_action = random.random() < self.epsilion
        self.epsilion = self.epsilion * self.epsilon_lr
        if random_action:
            return random.randint(0, ACTION_DIM-1), None  # allow model to learn that some actions are illegal
        return self.max_action(state_vector)

class RandomPokerModel():
    def next_action(self, state_vector, game):
        return random.choice(game.available_actions()), None
