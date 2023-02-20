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


class Model(nn.Module):
    """
    Define the SchrodingerNN,
    it consists of 5 hidden layers
    """

    def __init__(self, layer: int = 8, neurons: int = 40, act: str = 'tanh'):
        # Input layer
        super(Model, self).__init__()
        self.linear_in = nn.Linear(3, neurons)  # (x,y)
        # Output layer (5 for NS equation)
        self.linear_out = nn.Linear(neurons, 5)
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


def uv(model, x, y, t):
    model_out = model(torch.stack((x, y, t), dim=1))
    psi = model_out[:, 0:1]
    p = model_out[:, 1:2]
    s11 = model_out[:, 2:3]
    s22 = model_out[:, 3:4]
    s12 = model_out[:, 4:5]
    u = derivative(psi.squeeze(), y).unsqueeze(dim=1)
    v = -derivative(psi.squeeze(), x).unsqueeze(dim=1)
    return u, v, p, s11, s22, s12


def uv_uv(model, x, y, t):
    model_out = model(torch.stack((x, y, t), dim=1))
    psi = model_out[:, 0:1]
    u = derivative(psi.squeeze(), y).unsqueeze(dim=1)
    v = -derivative(psi.squeeze(), x).unsqueeze(dim=1)
    return u, v


def uv_uvp(model, x, y, t):
    model_out = model(torch.stack((x, y, t), dim=1))
    psi = model_out[:, 0:1]
    u = derivative(psi.squeeze(), y).unsqueeze(dim=1)
    v = -derivative(psi.squeeze(), x).unsqueeze(dim=1)
    p =  model_out[:, 1:2]
    return u, v, p


def uv_p(model, x, y, t):
    model_out = model(torch.stack((x, y, t), dim=1))
    p = model_out[:, 1:2]
    return p


def f(model, x_f, y_f, t_f, rho, mu):
    """
    This function evaluates the PDE at collocation points.
    """
    u, v, p, s11, s22, s12 = uv(model, x_f, y_f, t_f)

    s11_x = derivative(s11, x_f).unsqueeze(dim=1)
    s12_y = derivative(s12, y_f).unsqueeze(dim=1)
    s22_y = derivative(s22, y_f).unsqueeze(dim=1)
    s12_x = derivative(s12, x_f).unsqueeze(dim=1)

    # Plane stress problem
    u_x = derivative(u, x_f).unsqueeze(dim=1)
    u_y = derivative(u, y_f).unsqueeze(dim=1)

    v_x = derivative(v, x_f).unsqueeze(dim=1)
    v_y = derivative(v, y_f).unsqueeze(dim=1)

    # f_u = Sxx_x + Sxy_y
    f_u = rho * (u * u_x + v * u_y) - s11_x - s12_y
    f_v = rho * (u * v_x + v * v_y) - s12_x - s22_y

    # f_mass = u_x + v_y
    f_s11 = -p + 2 * mu * u_x - s11
    f_s22 = -p + 2 * mu * v_y - s22
    f_s12 = mu * (u_y + v_x) - s12

    f_p = p + (s11 + s22) / 2

    return f_u, f_v, f_s11, f_s22, f_s12, f_p


def mse_f(model, x_f, y_f, t_f):
    """
    This function calculates the MSE for the PDE.
    """
    f_u, f_v, f_s11, f_s22, f_s12, f_p = f(model, x_f, y_f, t_f, 1., 0.001)
    return [torch.mean(f_u ** 2), torch.mean(f_v ** 2), torch.mean(f_s11 ** 2), torch.mean(f_s22 ** 2), torch.mean(
        f_s12 ** 2), torch.mean(f_p ** 2)]


def mse_inlet(model, x_in, y_in, t_in, u_in, v_in):
    """
    This function calculates the MSE for the inlet.
    """
    u, v = uv_uv(model, x_in, y_in, t_in)
    return torch.mean((u - u_in.unsqueeze(dim=1)) ** 2) + torch.mean((v - v_in.unsqueeze(dim=1)) ** 2)


def mse_outlet(model, x_out, y_out, t_out):
    """
    This function calculates the MSE for the outlet.
    """
    p = uv_p(model, x_out, y_out, t_out)
    return torch.mean(p ** 2)


def mse_wall(model, x_wall, y_wall, t_wall):
    """
    This function calculates the MSE for the WALL area.
    """
    u, v = uv_uv(model, x_wall, y_wall, t_wall)
    return torch.mean(u ** 2) + torch.mean(v ** 2)

# def rel_error(model, pt_x, pt_y, pt_u):

def mse_ic(model, x_ic, y_ic, t_ic):
    u,v,p=uv_uvp(model, x_ic, y_ic, t_ic)
    return torch.mean(u**2)+torch.mean(v**2)+torch.mean(p**2)
