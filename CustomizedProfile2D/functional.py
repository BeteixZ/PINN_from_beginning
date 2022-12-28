"""
This file contains functions including
1. Random seed
2. Calculation of derivative
3. Weight initialization
4. Plotting and tensorboard function
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image

def set_seed(seed: int = 42):
    """
    Seeding the random variables for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    This function calculates the derivative of the model at x_f
    """
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x, grad_outputs=torch.ones_like(dy), create_graph=True, retain_graph=True)[0]
    return dy


def init_weights(m):
    """
    This function initializes the weights of the model by the normal Xavier initialization method.
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    pass


def args_summary(args):
    print(args)


def plot_t(model, device, iter, t_time, colorbar):
    x = np.arange(0, 1, 0.005)
    y = np.arange(0, 1, 0.005)
    ms_x, ms_y = np.meshgrid(x, y)
    # Just because meshgrid is used, we need to do the following adjustment
    x = np.ravel(ms_x).reshape(-1, 1)
    y = np.ravel(ms_y).reshape(-1, 1)
    t = np.zeros_like(y)+t_time

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=False).to(device)
    pt_y = Variable(torch.from_numpy(y).float(), requires_grad=False).to(device)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=False).to(device)
    pt_u = model((torch.stack((pt_x[:, 0], pt_y[:, 0], pt_t[:,0]), axis=1)))
    u = pt_u.data.cpu().numpy()
    ms_u = u.reshape(ms_x.shape)

    fig = plt.figure()
    plt.gca().set_aspect('equal')
    pc = plt.pcolormesh(ms_x, ms_y, ms_u, cmap=plt.get_cmap("rainbow"), linewidth=0, antialiased= True, )
    pc.set_clim(-1,1)
    plt.title('Iteration:'+str(iter)+'  Time:'+str(t_time))
    if colorbar:
        fig.colorbar(pc)
    return fig


def plot_slice(model, summary, device):
    plt.rcParams["figure.autolayout"] = True
    x = np.arange(0, 1, 0.01)
    t = np.arange(0, 1, 0.01)
    ms_x, ms_t = np.meshgrid(x, t)
    x = np.ravel(ms_x).reshape(-1, 1)
    t = np.ravel(ms_t).reshape(-1, 1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=False).to(device)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=False).to(device)
    pt_u = model((torch.stack((pt_x[:, 0], pt_t[:, 0]), axis=1)))
    u = pt_u.data.cpu().numpy()
    ms_u = u.reshape(ms_x.shape)


    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

    exact = np.exp(-ms_t)*np.sin(2*np.pi*ms_x)
    ax = plt.subplot(gs1[0, 0])
    ax.plot(ms_x[25,:], exact[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(ms_x[25,:], ms_u[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.25$', fontsize=15)
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-1, 1])
    plt.gca().set_aspect(0.5)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(ms_x[50,:], exact[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(ms_x[50,:], ms_u[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-1, 1])
    plt.gca().set_aspect(0.5)
    ax.set_title('$t = 0.50$', fontsize=15)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(ms_x[75,:], exact[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(ms_x[75,:], ms_u[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-1., 1.])
    plt.gca().set_aspect(0.5)
    ax.set_title('$t = 0.75$', fontsize=15)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    summary.add_figure('slice', fig)
    fig.savefig('./figures/slices.pdf', bbox_inches = 'tight', pad_inches=0.3)
