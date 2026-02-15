import torch.nn as nn

from agents.models.dqn_default_model import DQNDefaultModel
from agents.models.dqn_double_conv_model import DQNDoubleConv
from agents.models.dqn_mlp_model import DQNMlpModel
from agents.models.dqn_gnn_model import DQNGNNModel
from agents.models.ppo_default_model import PPODefaultModel
from agents.models.ppo_double_conv_model import PPODoubleConv
from agents.models.ppo_mlp_model import PPOMlpModel
from agents.models.ppo_gnn_model import PPOGNNModel
from agents.models.calculate_output_size import conv2d_size_out


def get_net(agent, net, activation, obs_space, act_space, h, w, **kwargs):
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(negative_slope=kwargs.get("negative_slope")),
        "swish": nn.SiLU()
    }

    other_nets = {
        "IDQN": {
            "default": lambda: DQNDefaultModel(
                obs_space=obs_space, 
                act_space=act_space, 
                h=h, 
                w=w, 
                activation=activations[activation]),
            "double_conv": lambda: DQNDoubleConv(
                obs_space=obs_space, 
                act_space=act_space, 
                h=conv2d_size_out(h), 
                w=conv2d_size_out(w), 
                activation=activations[activation]),
            "mlp": lambda: DQNMlpModel(
                obs_space=obs_space,
                act_space=act_space,
                h=h,
                w=w,
                activation=activations[activation]),
            "gnn": lambda: DQNGNNModel(
                obs_space=obs_space,
                act_space=act_space,
                h=h,
                w=w,
                activation=activations[activation])
        },
        "IPPO": {
            "default": lambda: PPODefaultModel(
                obs_space=obs_space, 
                act_space=act_space, 
                h=h,
                w=w, 
                activation=activations[activation]),
            "double_conv": lambda: PPODoubleConv(
                obs_space=obs_space,
                act_space=act_space,
                h=conv2d_size_out(h),
                w=conv2d_size_out(w),
                activation=activations[activation]),
            "mlp": lambda: PPOMlpModel(
                obs_space=obs_space,
                act_space=act_space,
                h=h,
                w=w,
                activation=activations[activation]),
            "gnn": lambda: PPOGNNModel(
                obs_space=obs_space,
                act_space=act_space,
                h=h,
                w=w,
                activation=activations[activation])
        }
    }
    return other_nets[agent][net]

