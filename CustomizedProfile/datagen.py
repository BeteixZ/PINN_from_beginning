import torch
import torch.nn as nn
import numpy as np
import scipy.io as sp
from functools import partial
from pyDOE import lhs
from torch.autograd import Variable
import time

from functional import set_seed


def initial_point(size, seed: int = 42):
    set_seed(seed)
    x_ic = np.random.uniform(low=0.0, high=1.0, size=size)
    t_ic = np.zeros_like(x_ic)
    u_ic = np.sin(2 * np.pi * x_ic)
    return x_ic, t_ic, u_ic


def bc_point(size, seed: int = 42):
    set_seed(seed)
    l_t_bc = np.random.uniform(low=0, high=1.0, size=size)
    u_t_bc = np.random.uniform(low=0, high=1.0, size=size)
    return l_t_bc, u_t_bc


def collocation_point(size, seed: int = 42):
    set_seed(seed)
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    c_f = lb + (ub - lb) * lhs(2, size)
    x_f = c_f[:, 0]
    t_f = c_f[:, 1]
    u_f = -np.exp(-t_f) * np.sin(2 * np.pi * x_f) + 4 * np.exp(-t_f) * np.pi * np.pi * np.sin(
        2 * np.pi * x_f)
    return x_f, t_f, u_f


def mesh_point():
    x = np.arange(0, 1, 0.01)
    t = np.arange(0, 1, 0.01)
    ms_x, ms_t = np.meshgrid(x, t)
    # Just because meshgrid is used, we need to do the following adjustment
    pt_x = np.ravel(ms_x).reshape(-1, 1)
    pt_t = np.ravel(ms_t).reshape(-1, 1)
    pt_u = np.exp(-pt_t[:,0])*np.sin(2*np.pi*pt_x[:,0])
    return pt_x, pt_t, pt_u