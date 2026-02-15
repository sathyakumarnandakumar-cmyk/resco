import torch.nn as nn
import pfrl
from pfrl.nn import Branched
from pfrl.policies import SoftmaxCategoricalHead

from agents.models.base_model import BaseModel


class PPOMlpModel(BaseModel):
    def __init__(self, obs_space, act_space, h, w, activation):
        super(PPOMlpModel, self).__init__(activation)

        # Input size: Full observation flattened (h,w passed in are conv2d-reduced, use obs_space directly)
        input_dim = obs_space[0] * obs_space[1] * obs_space[2]
        hidden_dim = 256

        # 5-Layer MLP with LayerNorm (mirrors DQN MLP architecture)
        self.fc1 = self.lecun_init(nn.Linear(input_dim, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = self.lecun_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = self.lecun_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.fc4 = self.lecun_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln4 = nn.LayerNorm(hidden_dim)

        # Actor-Critic heads
        self.policy_head = self.lecun_init(nn.Linear(hidden_dim, act_space), 1e-2)
        self.value_head = self.lecun_init(nn.Linear(hidden_dim, 1))
        self.branch = Branched(
            nn.Sequential(self.policy_head, SoftmaxCategoricalHead()),
            self.value_head
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.activation(self.ln3(self.fc3(x)))
        x = self.activation(self.ln4(self.fc4(x)))
        return self.branch(x)

    @staticmethod
    def lecun_init(layer, gain=1):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            pfrl.initializers.init_lecun_normal(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        return layer
