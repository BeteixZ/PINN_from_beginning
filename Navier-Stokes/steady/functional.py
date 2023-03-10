"""
This file contains functions including
1. Random seed
2. Calculation of derivative
3. Weight initialization
4. Plotting and tensorboard function
"""
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


def preprocess(dir='FenicsSol.mat'):
    '''
    Load reference solution from Fenics or Fluent
    '''
    data = scipy.io.loadmat(dir)

    X = data['x']
    Y = data['y']
    P = data['p']
    vx = data['vx']
    vy = data['vy']

    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    p_star = P.flatten()[:, None]
    vx_star = vx.flatten()[:, None]
    vy_star = vy.flatten()[:, None]

    return x_star, y_star, vx_star, vy_star, p_star


def postProcess(xmin, xmax, ymin, ymax, field_FLUENT, field_MIXED, s=2, alpha=0.5, marker='o'):
    [x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT] = field_FLUENT
    [x_MIXED, y_MIXED, u_MIXED, v_MIXED, p_MIXED] = field_MIXED

    fluent = pd.DataFrame(list(zip(x_FLUENT.flatten(), y_FLUENT.flatten(), u_FLUENT.flatten(), v_FLUENT.flatten(), p_FLUENT.flatten())))
    mixed = pd.DataFrame(list(zip(x_MIXED.flatten(), y_MIXED.flatten(), u_MIXED.flatten(), v_MIXED.flatten(), p_MIXED.flatten())))
    fluent.to_csv("fluent.csv")
    mixed.to_csv("mixed.csv")

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # Plot MIXED result
    cf = ax[0,0].scatter(x_MIXED[:, 0], y_MIXED[:, 0], c=u_MIXED[:, 0], alpha=alpha - 0.1, edgecolors='none',
                          cmap='rainbow', marker=marker, s=int(s))
    ax[0,0].axis('square')
    for key, spine in ax[0,0].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_xlim([xmin, xmax])
    ax[0,0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0,0].set_title(r'$u$ (m/s)')
    fig.colorbar(cf, ax=ax[0,0], fraction=0.046, pad=0.04)

    cf = ax[1,0].scatter(x_MIXED[:, 0], y_MIXED[:, 0], c=v_MIXED[:, 0], alpha=alpha - 0.1, edgecolors='none',
                          cmap='rainbow', marker=marker, s=int(s))
    ax[1,0].axis('square')
    for key, spine in ax[1,0].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_xlim([xmin, xmax])
    ax[1,0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1,0].set_title(r'$v$ (m/s)')
    fig.colorbar(cf, ax=ax[1,0], fraction=0.046, pad=0.04)

    cf = ax[2,0].scatter(x_MIXED[:, 0], y_MIXED[:, 0], c=p_MIXED[:, 0], alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker,
                          s=int(s), vmin=-0.25, vmax=4.0)
    ax[2,0].axis('square')
    for key, spine in ax[2,0].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[2,0].set_xticks([])
    ax[2,0].set_yticks([])
    ax[2,0].set_xlim([xmin, xmax])
    ax[2,0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2,0].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[2,0], fraction=0.046, pad=0.04)

    # Plot FLUENT result
    cf = ax[0, 1].scatter(x_FLUENT, y_FLUENT, c=u_FLUENT, alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker, s=s)
    ax[0, 1].axis('square')
    for key, spine in ax[0, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 1].set_title(r'$u$ (m/s)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)

    cf = ax[1, 1].scatter(x_FLUENT, y_FLUENT, c=v_FLUENT, alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker, s=s)
    ax[1, 1].axis('square')
    for key, spine in ax[1, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 1].set_title(r'$v$ (m/s)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)

    cf = ax[2, 1].scatter(x_FLUENT, y_FLUENT, c=p_FLUENT, alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker, s=s, vmin=-0.25, vmax=4.0)
    ax[2, 1].axis('square')
    for key, spine in ax[2, 1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    ax[2, 1].set_xlim([xmin, xmax])
    ax[2, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2, 1].set_title('Pressure (Pa)')
    fig.colorbar(cf, ax=ax[2, 1], fraction=0.046, pad=0.04)

    plt.savefig('./uvp.png', dpi=300)
    plt.close('all')
