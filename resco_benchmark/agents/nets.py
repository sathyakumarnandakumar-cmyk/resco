import torch.nn as nn
from agents.models.default import DefaultModel
from agents.models.double_conv import DoubleConv
from agents.models.calculate_output_size import conv2d_size_out
from agents.models.activations.swish import swish

def get_net(net, activation,    obs_space, act_space, h, w):
    activations = {
        'relu': nn.functional.relu,
        'tanh': nn.functional.tanh,
        'leaky_relu': nn.functional.leaky_relu,
        'swish': swish
    }
    h2 = conv2d_size_out(h)
    w2 = conv2d_size_out(w)
    other_nets = {
        "default" : DefaultModel(obs_space=obs_space,
                            act_space=act_space,
                            h=h,
                            w=w,
                            activation=activations[activation]),
        "double_conv": DoubleConv(
            obs_space=obs_space,
            act_space=act_space,
            h=h2,
            w=w2,
            activation=activations[activation]
        )
    }
    return other_nets[net]
    
    