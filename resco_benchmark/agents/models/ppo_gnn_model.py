import torch
import torch.nn as nn
import pfrl
from pfrl.nn import Branched
from pfrl.policies import SoftmaxCategoricalHead

from agents.models.base_model import BaseModel


class PPOGNNModel(BaseModel):
    def __init__(self, obs_space, act_space, h, w, activation):
        super(PPOGNNModel, self).__init__(activation)

        # h,w passed in are conv2d-reduced; use original obs_space dimensions
        self.num_nodes = obs_space[1]  # Number of lanes (original height)
        self.feature_dim = obs_space[0] * obs_space[2]  # Features per lane (channels * original width)

        # Hyperparameters
        self.hidden_dim = 128
        self.num_heads = 4

        # Input Projection
        self.input_proj = self.lecun_init(nn.Linear(self.feature_dim, self.hidden_dim))

        # Multi-Head Self-Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(self.hidden_dim)

        # Feed-Forward Network
        self.ff = nn.Sequential(
            self.lecun_init(nn.Linear(self.hidden_dim, self.hidden_dim * 2)),
            nn.ReLU(),
            self.lecun_init(nn.Linear(self.hidden_dim * 2, self.hidden_dim))
        )
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # Shared feature layer
        self.fc1 = self.lecun_init(nn.Linear(self.hidden_dim, 64))

        # Actor-Critic heads
        self.policy_head = self.lecun_init(nn.Linear(64, act_space), 1e-2)
        self.value_head = self.lecun_init(nn.Linear(64, 1))
        self.branch = Branched(
            nn.Sequential(self.policy_head, SoftmaxCategoricalHead()),
            self.value_head
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_nodes, -1)

        # 1. Input Projection
        x_proj = self.input_proj(x)

        # 2. Multi-Head Attention with Residual + Norm
        attn_out, _ = self.attention(x_proj, x_proj, x_proj)
        x = self.ln1(x_proj + attn_out)

        # 3. Feed Forward with Residual + Norm
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)

        # 4. Global Mean Pooling
        x = torch.mean(x, dim=1)

        # 5. Shared features then Actor-Critic heads
        x = self.activation(self.fc1(x))
        return self.branch(x)

    @staticmethod
    def lecun_init(layer, gain=1):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            pfrl.initializers.init_lecun_normal(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        return layer
