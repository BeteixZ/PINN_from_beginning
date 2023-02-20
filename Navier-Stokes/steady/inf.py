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
    args_summary, postProcess, preprocess
from model import Model, mse_f, mse_inlet, mse_outlet, mse_wall, uv
from datagen import ptsgen

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--layer', help='number of layers', type=int, default=8)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=40)
parser.add_argument('--act', help='activation function', type=str, default='mish')

def main():
    args = parser.parse_args()
    model = Model(args.layer, args.neurons, args.act).to(device)
    model.load_state_dict(torch.load("./models/new.pt"))
    model.eval()
    [x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT] = preprocess(dir='../FluentReferenceMu002/FluentSol.mat')
    field_FLUENT = [x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT]

    x_PINN = np.linspace(0, 1.1, 251)
    y_PINN = np.linspace(0, 0.41, 101)
    x_PINN, y_PINN = np.meshgrid(x_PINN, y_PINN)
    x_PINN = x_PINN.flatten()[:, None]
    y_PINN = y_PINN.flatten()[:, None]
    dst = ((x_PINN - 0.2) ** 2 + (y_PINN - 0.2) ** 2) ** 0.5
    x_PINN = x_PINN[dst >= 0.05]
    y_PINN = y_PINN[dst >= 0.05]
    x_PINN = x_PINN.flatten()[:, None]
    y_PINN = y_PINN.flatten()[:, None]
    x_PINN = Variable(torch.from_numpy(x_PINN.astype(np.float32)), requires_grad=True).to(device)
    y_PINN = Variable(torch.from_numpy(y_PINN.astype(np.float32)), requires_grad=True).to(device)

    u_PINN, v_PINN, p_PINN, _, _, _ = uv(model, x_PINN[:, 0], y_PINN[:, 0])
    x_PINN = x_PINN.data.cpu().numpy()
    y_PINN = y_PINN.data.cpu().numpy()
    u_PINN = u_PINN.data.cpu().numpy()
    u_PINN = np.reshape(u_PINN, (u_PINN.size, 1))
    v_PINN = v_PINN.data.cpu().numpy()
    v_PINN = np.reshape(v_PINN, (u_PINN.size, 1))
    p_PINN = p_PINN.data.cpu().numpy()
    field_MIXED = [x_PINN, y_PINN, u_PINN, v_PINN, p_PINN]

    postProcess(xmin=0, xmax=1.1, ymin=0, ymax=0.41, field_FLUENT=field_FLUENT, field_MIXED=field_MIXED, s=3,
                alpha=0.5)

if __name__ == "__main__":
    main()