"""
This file contains the model
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sp
from functools import partial
from pyDOE import lhs
from torch.autograd import Variable
import time
from torch import sin, exp
from numpy import pi

from functional import derivative


class Wave(nn.Module):
    """
    Define the SchrodingerNN,
    it consists of 5 hidden layers
    """

    def __init__(self, layer: int = 7, neurons: int = 50, act: str = 'tanh'):
        # Input layer
        super(Wave, self).__init__()
        self.linear_in = nn.Linear(3, neurons) # (x,y,t)
        # Output layer
        self.linear_out = nn.Linear(neurons, 1)
        # Hidden Layers
        self.layers = nn.ModuleList(
            [nn.Linear(neurons, neurons) for i in range(layer)]
        )
        # Activation function
        if act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'mish':
            self.act = nn.Mish()
        elif act == 'softplus':
            self.act = nn.Softplus()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x


def f(model, x_f, y_f, t_f):
    """
    This function evaluates the PDE at collocation points.
    """
    u = model(torch.stack((x_f, y_f, t_f), axis=1))[:, 0]  # Concatenates a seq of tensors along a new dimension
    u_t = derivative(u, t_f, order=1)
    u_xx = derivative(u, x_f, order=2)
    u_yy = derivative(u, y_f, order=2)
    u_f = ((8*pi**2)-1)*exp(-t_f)*sin(2*pi*x_f)*sin(2*pi*y_f)
    return u_t - u_xx - u_yy - u_f


def mse_f(model, x_f, y_f, t_f):
    """
    This function calculates the MSE for the PDE.
    """
    f_u = f(model, x_f, y_f, t_f)
    return (f_u ** 2).mean()


def mse_0(model, x_ic, y_ic, t_ic):
    """
    This function calculates the MSE for the initial condition.
    u_0 is the real values
    here u_ic should be sin(2pi x) sin(2pi y) defined in datagen
    """
    u = model(torch.stack((x_ic, y_ic , t_ic), axis=1))[:, 0]
    u_0 = sin(2*pi*x_ic)*sin(2*pi*y_ic)
    return ((u - u_0) ** 2).mean()


def mse_b(model, x_bc, y_bc, t_bc):
    """
    This function calculates the MSE for the boundary condition.
    """
    x_bc_diri = torch.zeros_like(y_bc)
    x_bc_diri.requires_grad = True
    y_bc_diri = torch.ones_like(x_bc)
    y_bc_diri.requires_grad = True
    u_bc_diri = torch.cat((model(torch.stack((x_bc_diri, y_bc, t_bc), axis=1))[:, 0],
                           model(torch.stack((x_bc, y_bc_diri, t_bc), axis=1))[:, 0]))
    mse_dirichlet = (u_bc_diri ** 2).mean()
    x_bc_nuem = torch.ones_like(y_bc)
    x_bc_nuem.requires_grad = True
    y_bc_nuem = torch.zeros_like(x_bc)
    y_bc_nuem.requires_grad = True
    u_bc_nuem_x = model(torch.stack((x_bc_nuem, y_bc, t_bc), axis=1))[:, 0]
    u_bc_nuem_y = model(torch.stack((x_bc, y_bc_nuem, t_bc), axis=1))[:, 0]
    u_x = derivative(u_bc_nuem_x, x_bc_nuem, 1)
    u_y = derivative(u_bc_nuem_y, y_bc_nuem, 1)
    u_x_0 = 2 * pi * exp(-t_bc) * sin(2 * pi * y_bc)
    u_y_0 = 2 * pi * exp(-t_bc) * sin(2 * pi * x_bc)
    mse_neumann = ((u_x - u_x_0) ** 2).mean() + ((u_y - u_y_0) ** 2).mean()
    return mse_dirichlet + mse_neumann


    #for i,t in enumerate(t_bc):
    #    t = torch.ones_like(x_bc[i]) * t
    #    x_bc_diri = torch.zeros_like(y_bc[i])
    #    x_bc_diri.requires_grad = True
    #    y_bc_diri = torch.ones_like(x_bc[i])
    #    y_bc_diri.requires_grad = True
    #    u_bc_diri = torch.cat((model(torch.stack((x_bc_diri, y_bc[i], t), axis=1))[:, 0],
    #                             model(torch.stack((x_bc[i], y_bc_diri, t), axis=1))[:, 0]))
    #    mse_dirichlet = (u_bc_diri ** 2).mean()
#
    #    x_bc_nuem = torch.ones_like(y_bc[i])
    #    x_bc_nuem.requires_grad = True
    #    y_bc_nuem = torch.zeros_like(x_bc[i])
    #    y_bc_nuem.requires_grad = True
    #    u_bc_nuem_x = model(torch.stack((x_bc_nuem, y_bc[i], t), axis = 1))[:,0]
    #    u_bc_nuem_y = model(torch.stack((x_bc[i], y_bc_nuem, t), axis = 1))[:,0]
    #    u_x = derivative(u_bc_nuem_x, x_bc_nuem, 1)
    #    u_y = derivative(u_bc_nuem_y, y_bc_nuem, 1)
    #    u_x_0 = 2*pi*exp(-t)*sin(2*pi*y_bc[i])
    #    u_y_0 = 2*pi*exp(-t)*sin(2*pi*x_bc[i])
    #    mse_neumann = ((u_x - u_x_0)**2).mean() + ((u_y-u_y_0)**2).mean()
#
    #    return mse_dirichlet + mse_neumann


def mse_data(model, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc):
    """
    miscellaneous
    """
    f_u = f(model, x_f, t_f, u_f)
    f_u_e = torch.zeros_like(f_u)
    u = model(torch.stack((x_ic, t_ic), axis=1))[:, 0]
    l_x_bc = torch.zeros_like(l_t_bc)
    l_x_bc.requires_grad = True
    l_u_bc = model(torch.stack((l_x_bc, l_t_bc), axis=1))[:, 0]
    u_x_bc = torch.ones_like(u_t_bc)
    u_x_bc.requires_grad = True
    u_u_bc = model(torch.stack((u_x_bc, u_t_bc), axis=1))[:, 0]
    u_x_b_upper = derivative(u_u_bc, u_x_bc, 1)
    pred_u = torch.concat((f_u, u, l_u_bc, u_x_b_upper))
    exact_u = torch.concat((f_u_e, u_ic, l_x_bc, 2 * pi * exp(-u_t_bc)))
    return pred_u, exact_u


def rel_error(model, pt_x, pt_y, pt_t, pt_u):
    u = (model(torch.stack((pt_x[:, 0], pt_y[:, 0], pt_t[:, 0]), axis=1))[:, 0]).detach().cpu().numpy()
    return np.linalg.norm(pt_u[:,0] - u) / np.linalg.norm(pt_u[:,0])
