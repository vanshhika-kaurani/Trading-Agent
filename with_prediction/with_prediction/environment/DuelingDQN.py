import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()

        # Feature extraction layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Dueling architecture: Separate Value and Advantage streams
        self.fc_value = nn.Linear(64, 32)
        self.value = nn.Linear(32, 1)

        self.fc_advantage = nn.Linear(64, 32)
        self.advantage = nn.Linear(32, action_dim)  # Output: 3 actions (buy, sell, hold)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = F.relu(self.fc_value(x))
        value = self.value(value)

        advantage = F.relu(self.fc_advantage(x))
        advantage = self.advantage(advantage)

        # Q-values: Q(s,a) = V(s) + A(s,a) - mean(A)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values