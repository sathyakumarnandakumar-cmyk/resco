import torch.nn as nn
import torch
import torch.nn.functional as F
from agents.models.base_model import BaseModel
from pfrl.q_functions import DiscreteActionValueHead

class DQNGNNModel(BaseModel):
    def __init__(self, obs_space, act_space, h, w, activation):
        super(DQNGNNModel, self).__init__(activation)
        
        self.num_nodes = h  # Number of lanes
        self.feature_dim = obs_space[0] * w # Features per lane
        
        # Hyperparameters
        self.hidden_dim = 128
        self.num_heads = 4  # Multi-Head Attention
        self.dropout = 0.1
        
        # Input Projection
        self.input_proj = nn.Linear(self.feature_dim, self.hidden_dim)
        
        # Multi-Head Self-Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        
        # Feed-Forward Network
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        
        # Output Head
        self.fc1 = nn.Linear(self.hidden_dim, 64)
        self.fc2 = nn.Linear(64, act_space)
        
        self.discrete_action_value_head = DiscreteActionValueHead()

    def forward(self, x):
        # x shape: [batch_size, 1, num_lanes, features]
        # Reshape to [batch_size, num_lanes, features]
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_nodes, -1)
        
        # 1. Input Projection
        x_proj = self.input_proj(x) # [B, N, H]
        
        # 2. Multi-Head Attention with Residual Connection & Norm
        attn_out, _ = self.attention(x_proj, x_proj, x_proj)
        x = self.ln1(x_proj + attn_out) # Add & Norm
        
        # 3. Feed Forward with Residual Connection & Norm
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out) # Add & Norm

        # 4. Global Mean Pooling (Aggregation across lanes)
        x = torch.mean(x, dim=1) # [B, H]
        
        # 5. Output Head
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return self.discrete_action_value_head(x)
