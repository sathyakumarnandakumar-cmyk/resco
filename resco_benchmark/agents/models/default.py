import torch
import torch.nn as nn
from pfrl.q_functions import DiscreteActionValueHead

from agents.models.activations.swish import SwishActivation


class DefaultModel(nn.Module):
    def __init__(self, obs_space, act_space, h, w):
        print(obs_space[0])
        super(DefaultModel, self).__init__()
        self.activation = SwishActivation()
        self.conv1 = nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(h * w * 64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_space)
        self.discrete_action_value_head = DiscreteActionValueHead()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return self.discrete_action_value_head(x)
