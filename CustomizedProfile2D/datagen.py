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
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    i_f = lb + (ub - lb) * lhs(2, size) # use Latin hyper sampling
    x_ic = i_f[:, 0]
    y_ic = i_f[:, 1]
    t_ic = np.zeros_like(x_ic)

    return x_ic, y_ic, t_ic


def bc_point(size, seed: int = 42):
    set_seed(seed)
    #lb = np.array([0.0, 0.0])
    #ub = np.array([1.0, 1.0])
    #x_bc = np.zeros((size, size))
    #y_bc = np.zeros((size, size))
    #for i in range(size):
    #    i_f = lb + (ub - lb) * lhs(2, size) # use Latin hyper sampling
    #    x_bc[i] = i_f[:, 0]
    #    y_bc[i] = i_f[:, 1]
    #t_bc = lhs(1, size)
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    c_f = lb + (ub - lb) * lhs(3, size)
    x_bc = c_f[:, 0]
    y_bc = c_f[:, 1]
    t_bc = c_f[:, 2]
    return x_bc, y_bc, t_bc


def collocation_point(size, seed: int = 42):
    set_seed(seed)
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    c_f = lb + (ub - lb) * lhs(3, size)
    x_f = c_f[:, 0]
    y_f = c_f[:, 1]
    t_f = c_f[:, 2]
    return x_f, y_f, t_f


def mesh_point():
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    t = np.arange(0, 1, 0.05)
    ms_x, ms_y, ms_t = np.meshgrid(x, y, t)
    # Just because meshgrid is used, we need to do the following adjustment
    pt_x = np.ravel(ms_x).reshape(-1, 1)
    pt_y = np.ravel(ms_y).reshape(-1, 1)
    pt_t = np.ravel(ms_t).reshape(-1, 1)
    pt_u = np.exp(-pt_t)*np.sin(2*np.pi*pt_x)*np.sin(2*np.pi*pt_y)
    return pt_x, pt_y, pt_t, pt_u
