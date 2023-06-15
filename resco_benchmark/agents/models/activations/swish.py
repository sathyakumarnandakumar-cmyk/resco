import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class SwishActivation(nn.Module):
    def __init__(self):
        super(SwishActivation, self).__init__()

    def forward(self, x):
        return swish(x)