from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self, activation):
        super(BaseModel, self).__init__()
        self.activation = activation
        self.flatten = nn.Flatten()

    @abstractmethod
    def forward(self, x):
        pass
