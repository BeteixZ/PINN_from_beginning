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
            dy, x, grad_outputs=torch.ones_like(dy), create_graph=True, retain_graph=True)[0]
    return dy


def initWeights(m):
    """
    This function initializes the weights of the models by the normal Xavier initialization method.
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.)
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


def plot_t_error(model, device, iter, t_time, colorbar):
    x = np.arange(0, 1, 0.005)
    y = np.arange(0, 1, 0.005)
    ms_x, ms_y = np.meshgrid(x, y)
    # Just because meshgrid is used, we need to do the following adjustment
    x = np.ravel(ms_x).reshape(-1, 1)
    y = np.ravel(ms_y).reshape(-1, 1)
    t = np.zeros_like(y)+t_time
    real_u = (np.exp(-t)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)).reshape(ms_x.shape)

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=False).to(device)
    pt_y = Variable(torch.from_numpy(y).float(), requires_grad=False).to(device)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=False).to(device)
    pt_u = model((torch.stack((pt_x[:, 0], pt_y[:, 0], pt_t[:,0]), axis=1)))
    u = pt_u.data.cpu().numpy()
    ms_u = u.reshape(ms_x.shape)


    fig = plt.figure()
    plt.gca().set_aspect('equal')
    pc = plt.pcolormesh(ms_x, ms_y, ms_u-real_u, cmap=plt.get_cmap("rainbow"), linewidth=0, antialiased= True, )
    pc.set_clim(-0.01, 0.01)
    plt.title('Iteration:'+str(iter)+'  Time:'+str(t_time))
    if colorbar:
        fig.colorbar(pc)
    return fig


def plot_slice(model, device):
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    ms_x, ms_y = np.meshgrid(x, y)
    time = np.arange(0, 1, 0.01)
    rel_error = []
    for t in time:
        t = np.ones_like(ms_x)*t
        pt_x = np.ravel(ms_x).reshape(-1, 1)
        pt_y = np.ravel(ms_y).reshape(-1, 1)
        pt_t = np.ravel(t).reshape(-1, 1)
        pt_u = np.exp(-pt_t)*np.sin(2*np.pi*pt_x)*np.sin(2*np.pi*pt_y)
        pt_x = Variable(torch.from_numpy(pt_x).float(), requires_grad=False).to(device)
        pt_y = Variable(torch.from_numpy(pt_y).float(), requires_grad=False).to(device)
        pt_t = Variable(torch.from_numpy(pt_t).float(), requires_grad=False).to(device)
        u = (model(torch.stack((pt_x[:, 0], pt_y[:, 0], pt_t[:,0]), axis=1))[:, 0]).detach().cpu().numpy()
        rel_error.append(np.linalg.norm(pt_u[:, 0] - u) / np.linalg.norm(pt_u[:, 0]))

    fig = plt.figure()
    #plt.gca().set_aspect('equal')
    plt.plot(time, rel_error)
    plt.xlabel('time')
    plt.ylabel('rel. error')
    return fig
