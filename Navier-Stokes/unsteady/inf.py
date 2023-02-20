import os
import shutil

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
import pandas as pd


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--layer', help='number of layers', type=int, default=8)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=40)
parser.add_argument('--act', help='activation function', type=str, default='tanh')

def main():
    args = parser.parse_args()
    model = Model(args.layer, args.neurons, args.act).to(device)
    model.load_state_dict(torch.load("./models/l8_n40_i200_b200_col30.0-lbfgs-tanh.pt"))
    model.eval()
    t_front = np.linspace(0, 0.5, 100)
    x_front = np.zeros_like(t_front)
    x_front.fill(0.15)
    y_front = np.zeros_like(t_front)
    y_front.fill(0.20)
    t_front = t_front.flatten()[:, None]
    x_front = x_front.flatten()[:, None]
    y_front = y_front.flatten()[:, None]

    x_frontT = Variable(torch.from_numpy(x_front.astype(np.float32)), requires_grad=True).to(device)
    y_frontT = Variable(torch.from_numpy(y_front.astype(np.float32)), requires_grad=True).to(device)
    t_frontT = Variable(torch.from_numpy(t_front.astype(np.float32)), requires_grad=True).to(device)

    model.eval()

    u_pred, v_pred, p_pred,_,_,_ = uv(model, x_frontT[:, 0], y_frontT[:, 0], t_frontT[:, 0])

    u_pred = u_pred.data.cpu().numpy()
    v_pred = v_pred.data.cpu().numpy()
    p_pred = p_pred.data.cpu().numpy()

    plt.plot(t_front, p_pred)
    plt.show()

    # Output u, v, p at each time step
    N_t = 51
    x_star = np.linspace(0, 1.1, 401)
    y_star = np.linspace(0, 0.41, 161)
    x_star, y_star = np.meshgrid(x_star, y_star)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    dst = ((x_star - 0.2) ** 2 + (y_star - 0.2) ** 2) ** 0.5
    x_star = x_star[dst >= 0.05]
    y_star = y_star[dst >= 0.05]
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]

    x_starT = Variable(torch.from_numpy(x_star.astype(np.float32)), requires_grad=True).to(device)
    y_starT = Variable(torch.from_numpy(y_star.astype(np.float32)), requires_grad=True).to(device)

    # shutil.rmtree('./output', ignore_errors=True)
    # os.makedirs('./output')

    fluent = pd.DataFrame()

    for i in range(N_t):
        t_star = np.zeros((x_star.size, 1))
        t_star.fill(i * 0.5 / (N_t - 1))

        t_starT = Variable(torch.from_numpy(t_star.astype(np.float32)), requires_grad=True).to(device)

        u_pred, v_pred, p_pred, _, _, _ = uv(model, x_starT[:, 0], y_starT[:, 0], t_starT[:, 0])
        u_pred = u_pred.data.cpu().numpy()
        v_pred = v_pred.data.cpu().numpy()
        p_pred = p_pred.data.cpu().numpy()
        amp_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5
        field = [x_star, y_star, t_star, u_pred, v_pred, p_pred, amp_pred]

        fluent = pd.concat([fluent,pd.DataFrame(list(
            zip(x_star.flatten(), y_star.flatten(), t_star.flatten(), u_pred.flatten(), v_pred.flatten())))])

        # postProcess(xmin=0, xmax=1.1, ymin=0, ymax=0.41, field=field, s=2, num=i)
        # make_gif()

    fluent.to_csv('full.csv')

if __name__ == '__main__':
    main()