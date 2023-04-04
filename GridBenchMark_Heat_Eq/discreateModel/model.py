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
from torch.optim.lr_scheduler import StepLR, SequentialLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from functional import derivative, initWeights, setSeed


class FCModel(nn.Module):
    def __init__(self, layer: int = 7, neurons: int = 40, q: int = None):
        super(FCModel, self).__init__()
        self.linear_in = nn.Linear(2, neurons)
        self.linear_out = nn.Linear(neurons, q)
        self.layers = nn.ModuleList(
            [nn.Linear(neurons, neurons) for i in range(layer)]
        )

        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x


class HeatEqModel:
    def __init__(self, nnPara, iterPara, q, pts=None, bound=None, tPara=None, save=None, record=None, record_name=None,
                 randSeed=127):
        setSeed(randSeed)
        self.device = "cpu"  # force to use GPU
        self.ptsCl = pts[0]
        self.ptsBc = pts[1]
        self.lowB = bound[0]
        self.uppB = bound[1]
        self.q = q
        self.tStart = tPara[0]
        self.tEnd = tPara[1]
        self.tDelta = (self.tEnd - self.tStart) / self.q
        self.model = FCModel(nnPara[0], nnPara[1], q + 1).to(self.device)

        self.iterADAM = iterPara[0]
        self.iterLBFGS = iterPara[1]
        self.nowIter = 0
        self.nowLoss = 0
        tmp = np.float32(np.loadtxt('./IRKWeights/Butcher_IRK%d.txt' % (q), ndmin=2))
        tmp1 = np.reshape(tmp[0: q ** 2 + q], (q + 1, q))
        self.IRKWeights = Variable(torch.from_numpy(tmp1.astype(np.float32)), requires_grad=True).to(self.device)
        self.IRKTimes = Variable(torch.from_numpy((tmp[q ** 2 + q: q ** 2 + 2 * q]* self.tEnd).astype(np.float32)),requires_grad=True).to(self.device)
        self.IRKTimesExtended = Variable(
            torch.from_numpy(np.append(tmp[q ** 2 + q:] * self.tEnd, self.tEnd).reshape(q + 1, 1).astype(np.float32)),
            requires_grad=True).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizerLFBGS = torch.optim.LBFGS(self.model.parameters(),
                                                max_iter=self.iterLBFGS,
                                                tolerance_change=0,
                                                line_search_fn="strong_wolfe")

        self.record = record
        self.recordName = record_name
        self.save = save
        self.writer = self.__summaryWriter(nnPara[0], nnPara[1])
        self.avgEpochTime = time.time()
        self.model.apply(initWeights)


    def __summaryWriter(self, layer, neurons):
        if self.record:
            return SummaryWriter(comment="")  # TODO

    def loadFCModel(self, dir):
        self.model.load_state_dict(torch.load(dir))

    def variablize(self, pts, reqGrad=None):
        if reqGrad is None:
            reqGrad = [True, True]
        colX = Variable(torch.from_numpy(pts[:, 0].astype(np.float32)), requires_grad=reqGrad[0]).to(self.device)
        colY = Variable(torch.from_numpy(pts[:, 1].astype(np.float32)), requires_grad=reqGrad[1]).to(self.device)
        return colX, colY

    def fwd_gradients_0(self, U, x):
        z = torch.ones(U.shape).to(self.device).requires_grad_(True)
        g = torch.autograd.grad(U, x, grad_outputs=z, create_graph=True)[0]
        return torch.autograd.grad(g, z, grad_outputs=torch.ones(g.shape).to(self.device), create_graph=True)[0]

    def fwd_gradients_1(self, U, x):
        z = torch.ones(U.shape).to(self.device).requires_grad_(True)
        g = torch.autograd.grad(U, x, grad_outputs=z, create_graph=True)[0]
        return torch.autograd.grad(g, z, grad_outputs=torch.ones(g.shape).to(self.device), create_graph=True)[0]

    def l2l1(self, x):
        # return torch.pow(torch.abs(x), self.nowIter / 5000 + 1)
        # return torch.pow(torch.abs(x), 2 - self.nowIter / 5000)
        return torch.square(x)

    def netU0(self):
        colX, colY = self.variablize(self.ptsCl)
        u1 = self.model(torch.stack((colX, colY), dim=1))
        u = u1[:, :-1]
        U_x = self.fwd_gradients_0(u, colX)
        u_xx = self.fwd_gradients_0(U_x, colX)
        u_y = self.fwd_gradients_0(u, colY)
        u_yy = self.fwd_gradients_0(u_y, colY)
        f = - u_xx - u_yy - (exp(-self.IRKTimes)*(-1+8*pi**2)*sin(2*pi*colX)*sin(2*pi*colY)).T
        u0 = u1 + self.tDelta * torch.matmul(f, self.IRKWeights.T)
        u0Real = (exp(-self.IRKTimesExtended) * sin(2 * pi * colX) * sin(2 * pi * colY)).T
        return torch.mean(self.l2l1(u0 - u0Real))

    def netU1(self):
        colX1, colY1 = self.variablize(self.ptsBc[0])
        colX2, colY2 = self.variablize(self.ptsBc[1])
        colX3, colY3 = self.variablize(self.ptsBc[2])
        colX4, colY4 = self.variablize(self.ptsBc[3])
        U1 = self.model(torch.stack((colX1, colY1), dim=1))
        U2 = self.model(torch.stack((colX2, colY2), dim=1))
        U3 = self.model(torch.stack((colX3, colY3), dim=1))
        U3_x = self.fwd_gradients_1(U3, colX3)
        U4 = self.model(torch.stack((colX4, colY4), dim=1))
        U4_y = self.fwd_gradients_1(U4, colY4)
        return torch.mean(self.l2l1(U1)) + torch.mean(self.l2l1(U2)) + \
                torch.mean(
                    self.l2l1(U3_x - (2 * pi * exp(-self.IRKTimesExtended) * sin(2 * pi * colY3)).T)) + \
                torch.mean(
                    self.l2l1(U4_y - (2 * pi * exp(-self.IRKTimesExtended) * sin(2 * pi * colX4)).T))

    def __closure(self):
        #with torch.profiler.profile(
        #        activities=[
        #            torch.profiler.ProfilerActivity.CPU,
        #            torch.profiler.ProfilerActivity.CUDA,
        #        ]
        #) as p:
        self.avgEpochTime = time.time()
        self.nowIter += 1
        self.optimizer.zero_grad()
        mseCl = self.netU0()
        mseBd = self.netU1()
        loss = mseCl + 1 * mseBd
        self.nowLoss = loss.item()
        loss.backward()
        if self.record:
            self.writer.add_scalar('Loss', loss.detach().cpu().numpy(), self.nowIter)
            self.writer.add_scalars('MSE', {'MSE_F': mseCl.detach().cpu().numpy(),
                                            'MSE_B': mseBd.detach().cpu().numpy()}, self.nowIter)

        if self.nowIter % 100 == 0:
                # self.__ptsTune()
            print("Iter: {}, AvgT: {:.2f}, loss: {:.6f}, F: {:.6f}, BD: {:.6f}"
                      .format(self.nowIter,
                              time.time() - self.avgEpochTime,
                              loss.item(),
                              mseCl,
                              mseBd))

        #print(p.key_averages().table(
        #            sort_by="self_cuda_time_total", row_limit=-1))

        return loss

    def train(self):
        timeStart = time.time()
        scheduler = SequentialLR(self.optimizer, [
            ExponentialLR(self.optimizer, gamma=1.0),
            ExponentialLR(self.optimizer, gamma=0.9995)
        ], milestones=[500])
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
                       './models/' + str(self.iterADAM + self.iterLBFGS) + "-" + self.recordName + '.pt')

    def inference(self):  # not change too much
        self.model.eval()

        ptsAll = np.concatenate((self.ptsCl, self.ptsBc[0], self.ptsBc[1], self.ptsBc[2], self.ptsBc[3]),
                                axis=0).astype(np.float32)
        solution = self.model(Variable(torch.from_numpy(ptsAll.astype(np.float32)), requires_grad=False).to(
            self.device)).detach().cpu().numpy().T
        realSolution = np.exp(-self.IRKTimesExtended.detach().cpu().numpy()) * np.sin(2 * pi * ptsAll[:, 0]) * np.sin(
            2 * pi * ptsAll[:, 1])

        # plot each slice of the solution by IRKTimesExtened

        def plotSolution():
            for t in range(self.q+1):
                plt.figure()
                plt.tricontourf(ptsAll[:, 0], ptsAll[:, 1], np.abs(solution[t, :] - realSolution[t, :]))
                plt.colorbar()
                plt.title("t = " + str(self.IRKTimesExtended[t].detach().cpu().numpy()))
                plt.savefig("./plots/" + str(t) + ".png")
                plt.close()

        plotSolution()

        absErr = np.abs(solution - realSolution)
        maxAbsErr = np.max(absErr)
        normedErr = np.linalg.norm(absErr) / np.linalg.norm(realSolution)

        print("Max error:", maxAbsErr, " Rel Error:", normedErr)
