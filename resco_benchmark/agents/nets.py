import torch.nn as nn
from pfrl.q_functions import DiscreteActionValueHead

from agents.models.default import DefaultModel
from agents.models.activations.swish import SwishActivation


def get_net(net, obs_space, act_space, h, w):
    nets = {
        "default": nn.Sequential(
            nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h * w * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_space),
            DiscreteActionValueHead(),
        ),
        "double_conv": nn.Sequential(
            nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(h * w * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_space),
            DiscreteActionValueHead(),
        ),
        "signle_conv_tanh": nn.Sequential(
            nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(h * w * 64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_space),
            DiscreteActionValueHead(),
        ),
        "signle_conv_leaky_relu": nn.Sequential(
            nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(h * w * 64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, act_space),
            DiscreteActionValueHead(),
        ),
        "default_swish": DefaultModel(obs_space=obs_space,
                            act_space=act_space,
                            h=h,
                            w=w,
                            acti=SwishActivation)
    }
    return nets[net]
