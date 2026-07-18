__all__ = ['ECGResNet']

import torch.nn as nn
from .xresnet1d import _xresnet1d


class ECGResNet(nn.Module):
    "ECGResNet"
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.xresnet1d = _xresnet1d(4, [3, 4, 6, 3], num_classes = self.num_classes) # ResNet 50 one-dimension
            
    def forward(self, input):
        return self.xresnet1d(input)