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


def preprocess(dir):
    # Directory of reference solution
    data = scipy.io.loadmat(dir)

    X = data['x']
    Y = data['y']
    Exact_u = data['u']
    Exact_v = data['v']
    Exact_p = data['p']

    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    u_star = Exact_u.flatten()[:, None]
    v_star = Exact_v.flatten()[:, None]
    p_star = Exact_p.flatten()[:, None]

    return x_star, y_star, u_star, v_star, p_star

def postProcess(xmin, xmax, ymin, ymax, field, s=2, num=0):
    ''' num: Number of time step
    '''
    print(num)
    [x_pred, y_pred, _, u_pred, v_pred, p_pred] = field

    # fig, axs = plt.subplots(2)
    fig, ax = plt.subplots(nrows=3, figsize=(6, 8))
    # fig.subplots_adjust(hspace=0.2, wspace=0.2)

    cf = ax[0].scatter(x_pred, y_pred, c=u_pred, alpha=0.7, edgecolors='none', cmap='rainbow', marker='o', s=s, vmin=0, vmax=1.4)
    ax[0].axis('square')
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title('u predict')
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)

    cf = ax[1].scatter(x_pred, y_pred, c=v_pred, alpha=0.7, edgecolors='none', cmap='rainbow', marker='o', s=s, vmin= -0.7,vmax=0.7)
    ax[1].axis('square')
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title('v predict')
    fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)

    cf = ax[2].scatter(x_pred, y_pred, c=p_pred, alpha=0.7, edgecolors='none', cmap='rainbow', marker='o', s=s, vmin=-0.2, vmax=3)
    ax[2].axis('square')
    ax[2].set_xlim([xmin, xmax])
    ax[2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2].set_title('p predict')
    fig.colorbar(cf, ax=ax[2], fraction=0.046, pad=0.04)

    # cf = ax[3].scatter(x_pred, y_pred, c=amp_pred, alpha=0.7, edgecolors='none', cmap='rainbow', marker='o', s=s,
    #                    vmin=-0.2, vmax=3)
    # ax[3].axis('square')
    # ax[3].set_xlim([xmin, xmax])
    # ax[3].set_ylim([ymin, ymax])
    # # cf.cmap.set_under('whitesmoke')
    # # cf.cmap.set_over('black')
    # ax[3].set_title('amp predict')
    # fig.colorbar(cf, ax=ax[3], fraction=0.046, pad=0.04)

    plt.suptitle('Time: '+str(num*0.01)+'s', fontsize=16)

    plt.savefig('./output/'+str(num)+'.png',dpi=150)
    plt.close('all')


def make_gif():
    frames = [Image.open(Path("./output/"+image)) for image in sorted(os.listdir("./output/"), key=lambda x: int(x[:-4]))[0:]]
    frame_one = frames[0]
    frame_one.save("./uv_with_time_full.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=1, include_color_table=True, optimize=False)