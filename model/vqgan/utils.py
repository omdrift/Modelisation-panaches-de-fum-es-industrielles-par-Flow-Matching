import torch
import torch.nn as nn
import math


def normalize(in_channels, **kwargs):
    return nn.GroupNorm(num_groups=math.gcd(32, in_channels), num_channels=in_channels, eps=1e-6, affine=True)


def swish(x):
    return x*torch.sigmoid(x)
