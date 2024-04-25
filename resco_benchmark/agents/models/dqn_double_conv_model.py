import torch.nn as nn

from agents.models.dqn_default_model import DQNDefaultModel


class DQNDoubleConv(DQNDefaultModel):
    def __init__(self, obs_space, act_space, h, w, activation):
        super(DQNDoubleConv, self).__init__(obs_space, act_space, h, w, activation)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(2, 2))

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return self.discrete_action_value_head(x)
