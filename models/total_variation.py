from __future__ import print_function, division
import os
import pathlib
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
from torchsummary import summary


def total_variation(gen_output, ideal_input):
    w_variance = torch.squeeze(torch.abs(gen_output[:, :, :, :-1] - gen_output[:, :, :, 1:]))
    v_variance = torch.squeeze(torch.abs(gen_output[:, :, :-1, :] - gen_output[:, :, 1:, :]))

    w_weight = torch.exp(-torch.abs(ideal_input[:, -1, :, :-1] - ideal_input[:, -1, :, 1:]))
    v_weight = torch.exp(-torch.abs(ideal_input[:, -1, :-1, :] - ideal_input[:, -1, 1:, :]))

    reduce_axes = (-3, -2, -1)
    x_var = (w_variance * w_weight).sum(dim=reduce_axes)
    y_var = (v_variance * v_weight).sum(dim=reduce_axes)

    return x_var + y_var


class TotalVariation(nn.Module):
    r"""Compute the Total Variation according to [1].

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N,)` or scalar.

    """

    def forward(self, gen_output, ideal_input) -> torch.Tensor:
        return total_variation(gen_output, ideal_input)