import torch.nn as nn
from pfrl.q_functions import DiscreteActionValueHead

from agents.models.base_model import BaseModel


class DQNDefaultModel(BaseModel):
    def __init__(self, obs_space, act_space, h, w, activation):
        super(DQNDefaultModel, self).__init__(activation)
        self.conv1 = nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2))
        self.fc1 = nn.Linear(h * w * 64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_space)
        self.discrete_action_value_head = DiscreteActionValueHead()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return self.discrete_action_value_head(x)
