import os
import shutil

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sp
from functools import partial

from matplotlib import pyplot as plt
from pyDOE import lhs
from torch.autograd import Variable
import time
from torch import sin, exp
from numpy import pi
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from functional import derivative, initWeights, setSeed, postProcess



class FCModel(nn.Module):
    def __init__(self, layer: int = 7, neurons: int = 40, act: str = 'tanh'):
        # Input layer
        super(FCModel, self).__init__()
        self.linear_in = nn.Linear(3, neurons)  # (x,y,t)
        self.linear_out = nn.Linear(neurons, 3)
        self.layers = nn.ModuleList(
            [nn.Linear(neurons, neurons) for i in range(layer)]
        )
        # Activation function
        if act == 'tanh':
            self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x


class HeatEqModel:
    def __init__(self, pts, bound, nnPara, iterPara, save, record, randSeed):
        setSeed(randSeed)
        self.device = "cuda:0"  # force to use GPU
        self.ptsCl = pts[0]
        self.ptsBc = pts[1]
        self.ptsIc = pts[2]
        self.lowB = bound[0]
        self.uppB = bound[1]
        self.model = FCModel(nnPara[0], nnPara[1], nnPara[2]).to(self.device)

        self.mseClV = 0
        self.mseInletV = 0
        self.mseOutletV = 0
        self.mseObsV = 0

        self.iterADAM = iterPara[0]
        self.iterLBFGS = iterPara[1]
        self.__nowIter = 0
        self.__nowLoss = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizerLFBGS = torch.optim.LBFGS(self.model.parameters(),
                                                max_iter=self.iterLBFGS,
                                                tolerance_change=0,
                                                line_search_fn="strong_wolfe")

        self.record = record
        self.save = save
        self.writer = self.__summaryWriter(nnPara[0], nnPara[1], nnPara[2])
        self.avgEpochTime = time.time()
        self.model.apply(initWeights)

    def __summaryWriter(self, layer, neurons, act):
        if self.record:
            return SummaryWriter(comment="")  # TODO

    def loadFCModel(self, dir):
        self.model.load_state_dict(torch.load(dir))

    def variablize(self, pts, reqGrad=None):
        if reqGrad is None:
            reqGrad = [True, True, True]
        colX = Variable(torch.from_numpy(pts[:, 0].astype(np.float32)), requires_grad=reqGrad[0]).to(self.device)
        colY = Variable(torch.from_numpy(pts[:, 1].astype(np.float32)), requires_grad=reqGrad[1]).to(self.device)
        colT = Variable(torch.from_numpy(pts[:, 2].astype(np.float32)), requires_grad=reqGrad[2]).to(self.device)
        return colX, colY, colT

    def __mseCollocation(self):
        colX, colY, colT = self.variablize(self.ptsCl, [True, True, True])
        u = self.model(torch.stack((colX, colY, colT),dim = 1))
        u_t = derivative(u, colT)
        u_xx = derivative(u, colX, order=2)
        u_yy = derivative(u, colT, order=2)
        u_f = ((8 * pi ** 2) - 1) * exp(-colT) * sin(2 * pi * colX) * sin(2 * pi * colY)
        return torch.square(u_t-u_xx-u_yy-u_f).mean()

    def __mseBoundary(self):
        # Dirichlet
        colX1, colY1, colT1 = self.variablize(self.ptsBc[0], [False, False, False])
        colX2, colY2, colT2 = self.variablize(self.ptsBc[1], [False, False, False])
        u1 = self.model(torch.stack((colX1, colY1, colT1), dim=1))
        u2 = self.model(torch.stack((colX2, colY2, colT2), dim=1))

        # neumann
        colX3, colY3, colT3 = self.variablize(self.ptsBc[0], [True, False, False])
        colX4, colY4, colT4 = self.variablize(self.ptsBc[1], [False, True, False])
        u3 = self.model(torch.stack((colX3, colY3, colT3), dim=1))
        u4 = self.model(torch.stack((colX4, colY4, colT4), dim=1))
        return torch.square(u1).mean() + torch.square(u2).mean() + \
               torch.square(u3 - 2*pi*exp(-colT3)*sin(2*pi*colY3)).mean() + \
                torch.square(u4 - 2*pi*exp(-colT4)*sin(2*pi*colT4)).mean()


    def __mseIC(self):
        colX, colY, colT = self.variablize(self.icPts, [False, False, False])
        u = self.model(torch.stack((colX, colY, colT), dim=1))
        return torch.mean(torch.square(u - sin(2*pi*colX)*sin(2*pi*colY)))

    def __closure(self):
        self.avgEpochTime = time.time()
        self.__nowIter += 1
        self.optimizer.zero_grad()
        mseCl = self.__mseCollocation()
        mseBd = self.__mseBoundary()
        mseIC = self.__mseIC()
        loss = mseCl + 1 * mseBd + 1 * mseIC
        self.__nowLoss = loss.item()
        loss.backward()
        if self.record:
            self.writer.add_scalar('Loss', loss.detach().cpu().numpy(), self.__nowIter)
            self.writer.add_scalars('MSE', {'MSE_F': mseCl.detach().cpu().numpy(),
                                            'MSE_B': mseBd.detach().cpu().numpy(),
                                            'MSE_IC': mseIC.detach().cpu().numpy()}, self.__nowIter)

        if self.__nowIter % 100 == 0:
            self.writer.add_scalars('weight', {'Alpha': self.alpha, 'Beta': self.beta}, self.__nowIter)
            print("Iter: {}, AvgT: {:.2f}, loss: {:.6f}, F: {:.6f}, BD: {:.6f}, IC:{:.6f}"
                  .format(self.__nowIter,
                          time.time() - self.avgEpochTime,
                          loss.item(),
                          mseCl,
                          mseBd,
                          mseIC,))
        return loss

    def train(self):
        timeStart = time.time()
        scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.5)
        print("Start training: ADAM")
        for i in range(self.iterADAM):
            self.optimizer.step(self.__closure)
            scheduler.step()

        torch.cuda.empty_cache()
        print("Start training: L-BFGS")
        self.optimizer = torch.optim.LBFGS(self.model.parameters(),
                                           lr=1,
                                           max_iter=self.iterLBFGS,
                                           history_size=100,
                                           tolerance_change=0,
                                           line_search_fn="strong_wolfe")
        self.optimizer.step(self.__closure)
        print('Total time cost: ', time.time() - timeStart, 's')

        if self.save:
            torch.save(self.model.state_dict(),
                       './models/' + "dortmund-2d-2-unsteay-models" + '.pt')

    def inference(self):  # not change too much
        self.model.eval()

        N_t = 51
        xStar = np.linspace(self.lowB[0], self.uppB[0], 401)
        yStar = np.linspace(self.lowB[1], self.uppB[1], 161)
        xStar, yStar = np.meshgrid(xStar, yStar)
        xStar = xStar.flatten()[:, None]
        yStar = yStar.flatten()[:, None]
        dst = ((xStar - 0.2) ** 2 + (yStar - 0.2) ** 2) ** 0.5  # consider change this: TODO
        xStar = xStar[dst >= 0.05]
        yStar = yStar[dst >= 0.05]

        xStarT = Variable(torch.from_numpy(xStar.astype(np.float32)), requires_grad=True).to(self.device)
        yStarT = Variable(torch.from_numpy(yStar.astype(np.float32)), requires_grad=True).to(self.device)

        shutil.rmtree('./output', ignore_errors=True)
        os.makedirs('./output')
        for i in range(N_t):
            tStar = np.zeros((xStar.size, 1))
            tStar.fill(i * self.uppB[2] / (N_t - 1))

            tStarT = Variable(torch.from_numpy(tStar.astype(np.float32)), requires_grad=False).to(self.device)

            modelOut = self.model(torch.stack((xStarT, yStarT, tStarT[:, 0]), dim=1))
            uPred = modelOut[:, 0].data.cpu().numpy()
            vPred = modelOut[:, 1].data.cpu().numpy()
            pPred = modelOut[:, 2].data.cpu().numpy()
            field = [xStar, yStar, tStar, uPred, vPred, pPred]

            postProcess(xmin=self.lowB[0], xmax=self.uppB[0], ymin=self.lowB[1], ymax=self.uppB[1], field=field, s=2,
                        num=i, tstep=self.uppB[2] / (N_t - 1))
