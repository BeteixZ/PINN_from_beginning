import os
import time
import random

from pyDOE import lhs
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class PINN_laminar_flow:
    def __init__(self, collocationPts, inletPts, outletPts, wallPts, layers, neurons, lowerBound, upperBound, dir):
        # Upper & lower bound
        self.lowerBound = lowerBound
        self.upperBound = upperBound

        # flow properties
        self.rho = 1.0
        self.mu = 0.02

        # collocation points
        self.collocationPtsX = collocationPts[:, 0:1]  # ((*,~),(*,~),...)
        self.collocationPtsY = collocationPts[:, 1:2]  # ((~,*),(~,*),...)

        # inlet points
        self.inletPtsX = inletPts[:, 0:1]  # ((*,~,~,~),(*,~,~,~),...)
        self.inletPtsY = inletPts[:, 1:2]  # ((~,*,~,~),(~,*,~,~),...)
        self.inletPtsU = inletPts[:, 2:3]  # ((~,~,*,~),(~,~,*,~),...)
        self.inletPtsV = inletPts[:, 3:4]  # ((~,~,~,*),(~,~,~,*),...)

        # outlet points
        self.outletPtsX = outletPts[:, 0:1]  # ((*,~),(*,~),...)
        self.outletPtsY = outletPts[:, 1:2]  # ((~,*),(~,*),...)

        # wall points
        self.wallPtsX = wallPts[:, 0:1]  # ((*,~),(*,~),...)
        self.wallPtsY = wallPts[:, 1:2]  # ((~,*),(~,*),...)

        # layers
        self.layers = layers
        self.neurons = neurons

        # direction
        self.dir = dir

        # torch variables
        self.learningRate = 0
        self.device = "cuda:0"
        self.collocationX = Variable(torch.from_numpy(self.collocationPtsX.astype(np.float32)), requires_grad=True).to(
            self.device)
        self.collocationY = Variable(torch.from_numpy(self.collocationPtsY.astype(np.float32)), requires_grad=True).to(
            self.device)
        self.inletX = Variable(torch.from_numpy(self.inletPtsX.astype(np.float32)), reqiures_grad=True).to(self.device)
        self.inletY = Variable(torch.from_numpy(self.inletPtsY.astype(np.float32)), requires_grad=True).to(self.device)
        self.inletU = Variable(torch.from_numpy(self.inletPtsU.astype(np.float32)), requires_grad=True).to(self.device)
        self.inletV = Variable(torch.from_numpy(self.inletPtsV.astype(np.float32)), requires_grad=True).to(self.device)
        self.outletX = Variable(torch.from_numpy(self.outletPtsX.astype(np.float32)), requires_grad=True).to(
            self.device)
        self.outletY = Variable(torch.from_numpy(self.outletPtsY.astype(np.float32)), requires_grad=True).to(
            self.device)
        self.wallX = Variable(torch.from_numpy(self.wallPtsX.astype(np.float32)), requires_grad=True).to(self.device)
        self.wallY = Variable(torch.from_numpy(self.wallPtsY.astype(np.float32)), requires_grad=True).to(self.device)

        #

        # loss

        # training parameter
        self.optimizer1 = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.optimizer2 = torch.optim.LBFGS(self.model.parameters(), lr=1,
                                  max_iter=epochs - 200,
                                  max_eval=epochs - 200,
                                  history_size=50,
                                  # tolerance_grad=0.01 * np.finfo(float).eps,
                                  tolerance_change=0,
                                  line_search_fn="strong_wolfe")

        # model
        self.model = Model(self.layers, self.neurons)
        self.epochs =

    def init_weights(m):
        """
        This function initializes the weights of the model by the normal Xavier initialization method.
        """
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
        pass


class Model(nn.Module):
    """
    Define the SchrodingerNN,
    it consists of 5 hidden layers
    """

    def __init__(self, layer: int = 7, neurons: int = 50):
        # Input layer
        super(Model, self).__init__()
        self.linear_in = nn.Linear(2, neurons)  # (x,y)
        # Output layer (5 for NS equation)
        self.linear_out = nn.Linear(neurons, 5)
        # Hidden Layers
        self.layers = nn.ModuleList(
            [nn.Linear(neurons, neurons) for i in range(layer)]
        )
        # Activation function
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x
