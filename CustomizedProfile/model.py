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

from functional import derivative


class Wave(nn.Module):
    """
    Define the SchrodingerNN,
    it consists of 5 hidden layers
    """

    def __init__(self, layer: int = 5, neurons: int = 20, act: str = 'tanh'):
        # Input layer
        super(Wave, self).__init__()
        self.linear_in = nn.Linear(2, neurons)
        # Output layer
        self.linear_out = nn.Linear(neurons, 1)
        # Hidden Layers
        self.layers = nn.ModuleList(
            [nn.Linear(neurons, neurons) for i in range(layer)]
        )
        # Activation function
        if act == 'tanh':
            self.act = nn.Tanh()  # How about LeakyReLU? Or even Swish?
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'Tanhshrink':
            self.act = nn.Tanhshrink()
        elif act == 'mish':
            self.act = nn.Mish()
        elif act == 'softplus':
            self.act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x


def seq_model():
   return torch.nn.Sequential(
       nn.Linear(2, 50),
       nn.Linear(50, 50),nn.ReLU(),
       nn.Linear(50, 50),nn.ReLU(),
       nn.Linear(50, 50),nn.ReLU(),
       nn.Linear(50, 50),nn.ReLU(),
       nn.Linear(50, 1)
   )

def f(model, x_f, t_f, u_f):
    """
    This function evaluates the PDE at collocation points.
    """
    u = model(torch.stack((x_f, t_f), axis=1))[:, 0]  # Concatenates a seq of tensors along a new dimension
    u_t = derivative(u, t_f, order=1)
    u_xx = derivative(u, x_f, order=2)
    return u_t - u_xx - u_f


def mse_f(model, x_f, t_f, u_f):
    """
    This function calculates the MSE for the PDE.
    """
    f_u = f(model, x_f, t_f, u_f)
    return (f_u ** 2).mean()


def mse_0(model, x_ic, t_ic, u_ic):
    """
    This function calculates the MSE for the initial condition.
    u_0 is the real values
    """
    u = model(torch.stack((x_ic, t_ic), axis=1))[:, 0]
    return ((u - u_ic) ** 2).mean()


def mse_b(model, l_t_bc, u_t_bc):
    """
    This function calculates the MSE for the boundary condition.
    """
    l_x_bc = torch.zeros_like(l_t_bc)
    l_x_bc.requires_grad = True
    l_u_bc = model(torch.stack((l_x_bc, l_t_bc), axis = 1))[:, 0]
    mse_dirichlet = (l_u_bc ** 2).mean()

    u_x_bc = torch.ones_like(u_t_bc)
    u_x_bc.requires_grad = True
    u_u_bc = model(torch.stack((u_x_bc, u_t_bc), axis=1))[:, 0]
    u_x_b_upper = derivative(u_u_bc, u_x_bc, 1)
    mse_neumann = (((2 * np.pi * torch.exp(-u_t_bc)) - u_x_b_upper) ** 2).mean()
    return mse_dirichlet + mse_neumann


def mse_data(model, x_f, t_f, u_f,x_ic, t_ic, u_ic, l_t_bc, u_t_bc):
    f_u = f(model, x_f, t_f, u_f)
    f_u_e = torch.zeros_like(f_u)
    u = model(torch.stack((x_ic, t_ic), axis=1))[:, 0]
    l_x_bc = torch.zeros_like(l_t_bc)
    l_x_bc.requires_grad = True
    l_u_bc = model(torch.stack((l_x_bc, l_t_bc), axis = 1))[:, 0]
    u_x_bc = torch.ones_like(u_t_bc)
    u_x_bc.requires_grad = True
    u_u_bc = model(torch.stack((u_x_bc, u_t_bc), axis=1))[:, 0]
    u_x_b_upper = derivative(u_u_bc, u_x_bc, 1)
    pred_u=torch.concat((f_u,u,l_u_bc,u_x_b_upper))
    exact_u=torch.concat((f_u_e,u_ic,l_x_bc,2 * np.pi * torch.exp(-u_t_bc)))
    return pred_u,exact_u

def rel_error(model, pt_x, pt_t, pt_u):
    u = (model(torch.stack((pt_x[:,0], pt_t[:,0]), axis=1))[:, 0]).detach().cpu().numpy()
    return np.linalg.norm(pt_u - u)/np.linalg.norm(pt_u)



