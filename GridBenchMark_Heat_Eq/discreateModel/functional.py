import os
from pathlib import Path

import scipy
import torch
import torch.nn as nn
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


def setSeed(seed: int = 42):
    """
    Seeding the random variables for reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    This function calculates the derivative of the models at x_f
    """
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    return dy



def initWeights(m):
    """
    This function initializes the weights of the models by the normal Xavier initialization method.
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.)
    pass



