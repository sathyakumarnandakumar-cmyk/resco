"""
PyTorch implementation of MA2C (Multi-Agent Advantage Actor-Critic).

Port of the TensorFlow MA2C to PyTorch, following the same agent API
as IDQN/IPPO (IndependentAgent with act/observe/set_mode).

Supports 4 model architectures via --net flag:
- mlp:         FC layers only (no recurrence)
- gru:         FC + GRU recurrent layer
- lstm:        FC + LSTM recurrent layer (original MA2C paper)
- transformer: FC + Self-Attention layer

Key features:
- Actor-critic network per junction
- Neighbor fingerprint exchange (action distributions shared between neighbors)
- On-policy learning with GAE-style advantage estimation
- RMSprop optimizer with gradient clipping
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent import Agent, IndependentAgent
from config.signal_config import signal_configs


# ============================================================================
# Model Registry
# ============================================================================

MA2C_MODELS = {}


def register_model(name):
    """Decorator to register a model class by name."""
    def decorator(cls):
        MA2C_MODELS[name] = cls
        return cls
    return decorator


# ============================================================================
# Top-level wrapper: IMA2C (Independent MA2C)
# ============================================================================

class IMA2C(IndependentAgent):
    """
    Multi-Agent Advantage Actor-Critic with independent agents per junction.
    Each agent shares its action distribution (fingerprint) with neighbors.
    """

    def __init__(self, config, obs_act, map_name, thread_number, **kwargs):
        super().__init__(config, obs_act, map_name, thread_number)
        self.signal_config = signal_configs[map_name]
        self.training = True
        net_type = kwargs.get('net', 'gru')

        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]

            # Calculate fingerprint size from neighbors
            downstream = self.signal_config[key]['downstream']
            neighbors = [downstream[direction] for direction in downstream]
            fp_size = 0
            for neighbor in neighbors:
                if neighbor is not None and neighbor in obs_act:
                    fp_size += obs_act[neighbor][1]  # neighbor's action size

            # Calculate wait lanes count
            lane_sets = self.signal_config[key]['lane_sets']
            lanes = []
            for direction in lane_sets:
                for lane in lane_sets[direction]:
                    if lane not in lanes:
                        lanes.append(lane)
            waits_len = len(lanes)

            self.agents[key] = MA2CAgent(
                config, obs_space, act_space, fp_size, waits_len,
                net_type=net_type,
                name='ma2c_' + key + '_' + str(thread_number)
            )

        if self.config.get('load', False):
            print('LOADING SAVED MODEL FOR EVALUATION')
            for key in self.agents:
                self.agents[key].load(config["models_for_visualization"] + key + ".pt")

    def _get_fingerprints(self, observation):
        """Collect fingerprints (action distributions) from neighboring agents."""
        agent_fingerprint = {}
        for agent_id in observation.keys():
            downstream = self.signal_config[agent_id]['downstream']
            neighbors = [downstream[direction] for direction in downstream]
            fingerprints = []
            for neighbor in neighbors:
                if neighbor is not None and neighbor in self.agents:
                    neighbor_fp = self.agents[neighbor].fingerprint
                    fingerprints.append(neighbor_fp)
            if len(fingerprints) > 0:
                agent_fingerprint[agent_id] = np.concatenate(fingerprints)
            else:
                agent_fingerprint[agent_id] = np.array([], dtype=np.float32)
        return agent_fingerprint

    def act(self, observation):
        acts = {}
        fingerprints = self._get_fingerprints(observation)
        for agent_id in observation.keys():
            env_obs = observation[agent_id]
            neighbor_fps = fingerprints[agent_id]
            combined = np.concatenate([env_obs, neighbor_fps])
            acts[agent_id] = self.agents[agent_id].act(combined)
        return acts

    def observe(self, observation, reward, done, info):
        fingerprints = self._get_fingerprints(observation)
        for agent_id in observation.keys():
            env_obs = observation[agent_id]
            neighbor_fps = fingerprints[agent_id]
            combined = np.concatenate([env_obs, neighbor_fps])
            self.agents[agent_id].observe(combined, reward[agent_id], done, info)

    def set_mode(self, mode):
        self.training = (mode == "training")
        for agent in self.agents.values():
            agent.set_mode(mode)


# ============================================================================
# Per-junction agent: MA2CAgent
# ============================================================================

class MA2CAgent(Agent):
    """Single-junction MA2C agent with configurable network architecture."""

    def __init__(self, config, observation_shape, num_actions, fingerprint_size,
                 waits_len, net_type='gru', name=''):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.name = name

        self.steps_done = 0
        self.state = None
        self.value = None
        self.action = None
        self.fingerprint = np.zeros(num_actions, dtype=np.float32)
        self.training = True

        # Observation is 1D: [wave_features, wait_features, fingerprint_features]
        n_s = observation_shape[0] + fingerprint_size
        n_a = num_actions
        n_w = waits_len
        n_f = fingerprint_size

        num_hidden = config.get('num_hidden', 128)
        num_recurrent = config.get('num_recurrent', 64)
        batch_size = config.get('batch_size', 120)

        print(f'[MA2C] {name}: n_s={n_s}, n_a={n_a}, n_w={n_w}, n_f={n_f}, net={net_type}')

        # Build the network from registry
        if net_type not in MA2C_MODELS:
            raise ValueError(f"Unknown net type '{net_type}'. "
                             f"Available: {list(MA2C_MODELS.keys())}")
        self.model = MA2C_MODELS[net_type](
            n_s=n_s, n_a=n_a, n_w=n_w, n_f=n_f,
            num_hidden=num_hidden, num_recurrent=num_recurrent
        ).to(self.device)

        lr = config.get('lr_init', 2.5e-4)
        alpha = config.get('rmsp_alpha', 0.99)
        epsilon = config.get('rmsp_epsilon', 1e-5)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=lr, alpha=alpha, eps=epsilon
        )

        self.gamma = config.get('gamma', 0.96)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 40)
        self.reward_norm = config.get('reward_norm', 2000.0)
        self.reward_clip = config.get('reward_clip', 2.0)
        self.batch_size = batch_size

        self.buffer = OnPolicyBuffer(self.gamma)

    def act(self, observation):
        self.state = observation

        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(obs_tensor)
            policy = policy.squeeze(0).cpu().numpy()
            self.value = value.squeeze(0).item()

        # Ensure valid probability distribution
        policy = np.clip(policy, 1e-8, 1.0)
        policy = policy / policy.sum()

        if self.training:
            self.action = np.random.choice(np.arange(self.num_actions), p=policy)
        else:
            self.action = np.argmax(policy)

        self.fingerprint = policy.copy()
        return self.action

    def observe(self, observation, reward, done, info):
        if not self.training:
            if done:
                self.model.reset_hidden()
            return

        # Normalize and clip reward
        if self.reward_norm:
            reward = reward / self.reward_norm
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)

        self.buffer.add_transition(self.state, self.action, reward, self.value, done)
        self.steps_done += 1

        # Backward pass every batch_size steps or on episode end
        if self.steps_done % self.batch_size == 0 or done:
            if done:
                R = 0.0
            else:
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, value = self.model(obs_tensor)
                    R = value.squeeze(0).item()
            self._backward(R)

        if done:
            self.steps_done = 0
            self.model.reset_hidden()
            self.buffer.reset()

    def _backward(self, R):
        """Compute loss and update network weights."""
        obs, acts, dones, Rs, Advs = self.buffer.sample_transition(R)

        if len(obs) == 0:
            return

        obs_t = torch.FloatTensor(obs).to(self.device)
        acts_t = torch.LongTensor(acts).to(self.device)
        Rs_t = torch.FloatTensor(Rs).to(self.device)
        Advs_t = torch.FloatTensor(Advs).to(self.device)

        # Forward pass (recompute with gradient tracking)
        self.model.reset_hidden()
        policies = []
        values = []
        for i in range(len(obs)):
            pi, v = self.model(obs_t[i:i+1])
            policies.append(pi)
            values.append(v)
        policies = torch.cat(policies, dim=0)   # (T, n_a)
        values = torch.cat(values, dim=0).squeeze(-1)    # (T,)

        # Policy loss
        log_pi = torch.log(torch.clamp(policies, 1e-10, 1.0))
        action_one_hot = F.one_hot(acts_t, self.num_actions).float()
        selected_log_pi = (log_pi * action_one_hot).sum(dim=1)
        policy_loss = -(selected_log_pi * Advs_t).mean()

        # Entropy loss (encourages exploration)
        entropy = -(policies * log_pi).sum(dim=1)
        entropy_loss = -entropy.mean() * self.entropy_coef

        # Value loss
        value_loss = F.mse_loss(values, Rs_t) * 0.5 * self.value_coef

        # Total loss
        loss = policy_loss + value_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path + '.pt')

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def set_mode(self, mode):
        if mode == "training":
            self.training = True
            self.model.train()
        elif mode == "validation":
            self.training = False
            self.model.eval()
        else:
            raise ValueError(f"Unsupported agent mode: {mode}")


# ============================================================================
# Shared Feature Encoder (used by all models)
# ============================================================================

class FeatureEncoder(nn.Module):
    """
    Shared feature encoder: separate FC layers for wave, wait, and fingerprint
    features, concatenated into a single representation.
    """

    def __init__(self, n_s, n_w, n_f, num_hidden):
        super().__init__()
        self.n_s = n_s
        self.n_w = n_w
        self.n_f = n_f

        wave_size = n_s - n_w - n_f
        self.fc_wave = nn.Linear(max(wave_size, 1), num_hidden)

        self.fc_wait = nn.Linear(n_w, num_hidden // 4) if n_w > 0 else None
        self.fc_fp = nn.Linear(n_f, num_hidden // 4) if n_f > 0 else None

        self.output_size = num_hidden
        if n_w > 0:
            self.output_size += num_hidden // 4
        if n_f > 0:
            self.output_size += num_hidden // 4

    def forward(self, x):
        wave_end = self.n_s - self.n_w - self.n_f
        h_wave = F.relu(self.fc_wave(x[:, :wave_end]))
        features = [h_wave]

        if self.fc_wait is not None and self.n_w > 0:
            h_wait = F.relu(self.fc_wait(x[:, wave_end:wave_end + self.n_w]))
            features.append(h_wait)
        if self.fc_fp is not None and self.n_f > 0:
            h_fp = F.relu(self.fc_fp(x[:, wave_end + self.n_w:]))
            features.append(h_fp)

        return torch.cat(features, dim=-1)


# ============================================================================
# Model 1: MLP Actor-Critic (no recurrence)
# ============================================================================

@register_model('mlp')
class MLPActorCritic(nn.Module):
    """
    Simple MLP actor-critic — no temporal memory.
    FC encoder → LayerNorm → ReLU → policy/value heads
    """

    def __init__(self, n_s, n_a, n_w, n_f, num_hidden=128, num_recurrent=64):
        super().__init__()
        self.encoder = FeatureEncoder(n_s, n_w, n_f, num_hidden)
        enc_out = self.encoder.output_size

        self.fc = nn.Linear(enc_out, num_recurrent)
        self.ln = nn.LayerNorm(num_recurrent)
        self.policy_head = nn.Linear(num_recurrent, n_a)
        self.value_head = nn.Linear(num_recurrent, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def reset_hidden(self):
        pass  # No recurrent state

    def forward(self, x):
        h = self.encoder(x)
        h = F.relu(self.ln(self.fc(h)))
        policy = F.softmax(self.policy_head(h), dim=-1)
        value = self.value_head(h)
        return policy, value


# ============================================================================
# Model 2: GRU Actor-Critic
# ============================================================================

@register_model('gru')
class GRUActorCritic(nn.Module):
    """
    GRU-based actor-critic — lightweight recurrence.
    FC encoder → GRUCell → policy/value heads
    """

    def __init__(self, n_s, n_a, n_w, n_f, num_hidden=128, num_recurrent=64):
        super().__init__()
        self.num_recurrent = num_recurrent
        self.encoder = FeatureEncoder(n_s, n_w, n_f, num_hidden)
        enc_out = self.encoder.output_size

        self.gru = nn.GRUCell(enc_out, num_recurrent)
        self.policy_head = nn.Linear(num_recurrent, n_a)
        self.value_head = nn.Linear(num_recurrent, 1)
        self.hidden = None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x):
        batch_size = x.size(0)
        h = self.encoder(x)

        if self.hidden is None:
            self.hidden = torch.zeros(batch_size, self.num_recurrent, device=x.device)
        self.hidden = self.gru(h, self.hidden)
        self.hidden = self.hidden.detach()

        policy = F.softmax(self.policy_head(self.hidden), dim=-1)
        value = self.value_head(self.hidden)
        return policy, value


# ============================================================================
# Model 3: LSTM Actor-Critic (original MA2C paper)
# ============================================================================

@register_model('lstm')
class LSTMActorCritic(nn.Module):
    """
    LSTM-based actor-critic — matches original MA2C paper architecture.
    FC encoder → LSTMCell → policy/value heads
    """

    def __init__(self, n_s, n_a, n_w, n_f, num_hidden=128, num_recurrent=64):
        super().__init__()
        self.num_recurrent = num_recurrent
        self.encoder = FeatureEncoder(n_s, n_w, n_f, num_hidden)
        enc_out = self.encoder.output_size

        self.lstm = nn.LSTMCell(enc_out, num_recurrent)
        self.policy_head = nn.Linear(num_recurrent, n_a)
        self.value_head = nn.Linear(num_recurrent, 1)
        self.hidden = None  # (h, c) tuple

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x):
        batch_size = x.size(0)
        h = self.encoder(x)

        if self.hidden is None:
            h0 = torch.zeros(batch_size, self.num_recurrent, device=x.device)
            c0 = torch.zeros(batch_size, self.num_recurrent, device=x.device)
            self.hidden = (h0, c0)

        h_new, c_new = self.lstm(h, self.hidden)
        self.hidden = (h_new.detach(), c_new.detach())

        policy = F.softmax(self.policy_head(h_new), dim=-1)
        value = self.value_head(h_new)
        return policy, value


# ============================================================================
# Model 4: Transformer Actor-Critic
# ============================================================================

@register_model('transformer')
class TransformerActorCritic(nn.Module):
    """
    Transformer-based actor-critic — captures long-range temporal dependencies.
    FC encoder → buffer recent observations → TransformerEncoder → policy/value heads

    Uses a sliding window of recent observations as the sequence for
    self-attention, providing temporal context without explicit recurrence.
    """

    def __init__(self, n_s, n_a, n_w, n_f, num_hidden=128, num_recurrent=64,
                 nhead=4, num_layers=2, seq_len=16):
        super().__init__()
        self.seq_len = seq_len
        self.num_recurrent = num_recurrent

        self.encoder = FeatureEncoder(n_s, n_w, n_f, num_hidden)
        enc_out = self.encoder.output_size

        # Project to transformer dimension (must be divisible by nhead)
        d_model = num_recurrent
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = ((d_model // nhead) + 1) * nhead
        self.d_model = d_model

        self.input_proj = nn.Linear(enc_out, d_model)

        # Positional encoding (learned)
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.policy_head = nn.Linear(d_model, n_a)
        self.value_head = nn.Linear(d_model, 1)

        # Observation history buffer (sliding window)
        self.obs_buffer = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def reset_hidden(self):
        self.obs_buffer = None

    def _generate_causal_mask(self, sz, device):
        """Generate causal attention mask (lower triangular)."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, x):
        batch_size = x.size(0)
        h = self.encoder(x)
        h = self.input_proj(h)  # (batch, d_model)

        # Add to observation buffer
        if self.obs_buffer is None:
            self.obs_buffer = h.unsqueeze(1)  # (batch, 1, d_model)
        else:
            self.obs_buffer = torch.cat([self.obs_buffer.detach(), h.unsqueeze(1)], dim=1)
            if self.obs_buffer.size(1) > self.seq_len:
                self.obs_buffer = self.obs_buffer[:, -self.seq_len:]

        seq_len = self.obs_buffer.size(1)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        seq = self.obs_buffer + pos_emb

        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, x.device)

        # Transformer forward
        out = self.transformer(seq, mask=causal_mask)

        # Use the last token's output
        last_out = out[:, -1, :]  # (batch, d_model)

        policy = F.softmax(self.policy_head(last_out), dim=-1)
        value = self.value_head(last_out)
        return policy, value


# ============================================================================
# On-Policy Replay Buffer
# ============================================================================

class OnPolicyBuffer:
    """
    On-policy transition buffer that computes returns and advantages.
    Ported from the TF MA2C implementation.
    """

    def __init__(self, gamma):
        self.gamma = gamma
        self.reset()

    def reset(self, done=False):
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.dones = [done]

    def add_transition(self, ob, a, r, v, done):
        self.obs.append(ob)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def sample_transition(self, R):
        """Compute returns and advantages, then return the batch."""
        if len(self.obs) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        Rs = []
        Advs = []
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.0 - float(done))
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()

        obs = np.array(self.obs, dtype=np.float32)
        acts = np.array(self.acts, dtype=np.int32)
        Rs = np.array(Rs, dtype=np.float32)
        Advs = np.array(Advs, dtype=np.float32)

        # Reset buffer but keep the last done state
        self.reset(self.dones[-1])

        return obs, acts, None, Rs, Advs
