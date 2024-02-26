import torch.nn as nn
import pfrl
from pfrl.nn import Branched
from pfrl.policies import SoftmaxCategoricalHead

from agents.models.base_model import BaseModel


class PPODefaultModel(BaseModel):
    def __init__(self, obs_space, act_space, h, w, activation):
        super(PPODefaultModel, self).__init__(activation)
        self.conv1 = self.lecun_init(nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)))
        self.fc1 = self.lecun_init(nn.Linear(h*w*64, 64))
        self.fc2 = self.lecun_init(nn.Linear(64, 64))
        self.fc3 = self.lecun_init(nn.Linear(64, act_space), 1e-2)
        self.fc4 = self.lecun_init(nn.Linear(64, 1))
        self.branch = Branched(
            nn.Sequential(self.fc3, SoftmaxCategoricalHead()),
            self.fc4
        )

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.branch(x)

    @staticmethod
    def lecun_init(layer, gain=1):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            pfrl.initializers.init_lecun_normal(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        else:
            pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
            pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)
        return layer
