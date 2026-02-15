import torch.nn as nn
import torch
import torch.nn.functional as F
from agents.models.base_model import BaseModel
from pfrl.q_functions import DiscreteActionValueHead

class DQNMlpModel(BaseModel):
    def __init__(self, obs_space, act_space, h, w, activation):
        super(DQNMlpModel, self).__init__(activation)
        
        # Input size: Flattened vector of (num_lanes * features_per_lane)
        input_dim = obs_space[0] * h * w
        hidden_dim = 256  # Increased from 64 to 256 (Wider)

        # 5-Layer MLP (Deeper) with LayerNorm
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

        self.fc5 = nn.Linear(hidden_dim, act_space)
        
        self.discrete_action_value_head = DiscreteActionValueHead()

    def forward(self, x):
        # x shape: [batch_size, channels, h, w]
        # Flatten: [batch_size, channels * h * w]
        x = x.view(x.size(0), -1)
        
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.activation(self.ln3(self.fc3(x)))
        x = self.activation(self.ln4(self.fc4(x)))
        
        x = self.fc5(x)
        
        return self.discrete_action_value_head(x)
