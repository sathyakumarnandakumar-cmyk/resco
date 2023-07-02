import torch.nn as nn
from pfrl.q_functions import DiscreteActionValueHead
from agents.models.default import DefaultModel
from agents.models.calculate_output_size import conv2d_size_out


def get_net(net, obs_space, act_space, h, w):
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'leaky_relu': nn.LeakyReLU,
        'swish': nn.SiLU
    }
    h2 = conv2d_size_out(h)
    w2 = conv2d_size_out(w)
    other_nets = {
        "double_conv": nn.Sequential(
            nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h2 * w2 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_space),
            DiscreteActionValueHead(),
        ),
    }
    if net[:7] == 'default':
        return DefaultModel(obs_space=obs_space,
                            act_space=act_space,
                            h=h,
                            w=w,
                            activation=activations[net[8:]]())
    else:
        return other_nets[net]
    
    