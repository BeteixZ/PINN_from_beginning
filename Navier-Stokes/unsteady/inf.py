import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from torch.autograd import Variable
import time
import argparse

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from functional import set_seed, init_weights, \
    args_summary, postProcess, preprocess, make_gif
from model import Model, mse_f, mse_inlet, mse_outlet, mse_wall, uv
from datagen import ptsgen


def main():
    make_gif()


if __name__ == '__main__':
    main()