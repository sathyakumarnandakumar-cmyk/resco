import torch.nn as nn

from agents.models.ppo_default_model import PPODefaultModel


class PPODoubleConv(PPODefaultModel):
    def __init__(self, obs_space, act_space, h, w, activation):
        super(PPODoubleConv, self).__init__(obs_space, act_space, h, w, activation)
        self.conv2 = self.lecun_init(nn.Conv2d(64, 64, kernel_size=(2, 2)))

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.branch(x)
