"""
This file contains the main code for PINN including adjustable
1. Neural Layer
2. Neurons
3. Points (Init, Bc and Collo.)
4. Epochs
5. LR

Every run writes to TensorBoard file.、
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sp
from functools import partial
from pyDOE import lhs
from torch.autograd import Variable
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

from functional import set_seed, init_weights, args_summary, plot
from model import Wave, mse_f, mse_0, mse_b, rel_error
from datagen import initial_point, bc_point, collocation_point, mesh_point


seed = 42

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# parser = argparse.ArgumentParser()
# parser.add_argument('--layer', help='number of layers', type=int, default=5)
# parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=20)
# parser.add_argument('--initpts', help='number of init points pper layer', type=int, default=50)
# parser.add_argument('--bcpts', help='number of boundary points', type=int, default=50)
# parser.add_argument('--colpts', help='number of collocation points', type=int, default=10000)
# parser.add_argument('--epochs', help='number of epochs', type=int, default=1500)
# parser.add_argument('--lr', help='learning rate', type=float, default=1)


def closure(model, optimizer, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, summary):
    """
    The closure function to use L-BFGS optimization method.
    """
    global iter
    global final_loss
    global final_relerr
    optimizer.zero_grad()
    # evaluating the MSE for the PDE
    msef = mse_f(model, x_f, t_f, u_f)
    mse0 = mse_0(model, x_ic, t_ic, u_ic)
    mseb = mse_b(model, l_t_bc, u_t_bc)
    loss = msef + mse0 + mseb
    pt_x, pt_t, pt_u = mesh_point()
    pt_x = Variable(torch.from_numpy(pt_x).float(), requires_grad=False).to(device)
    pt_t = Variable(torch.from_numpy(pt_t).float(), requires_grad=False).to(device)
    relerror = rel_error(model, pt_x, pt_t, pt_u)
    summary.add_scalar('Loss', loss, iter)
    summary.add_scalars('MSE', {'MSE_f': msef, 'MSE_init': mse0, 'MSE_bc': mseb}, iter)
    summary.add_scalar('MSE_relerror', relerror, iter)
    loss.backward(retain_graph=True)
    iter += 1
    final_loss = loss
    final_relerr = relerror
    if iter % 1000 == 0:
        print(f"[Iter: {iter}] loss: {loss.item()}, msef:{msef}, mse0:{mse0}, mseb:{mseb}, relerr:{relerror}")
    # if iter % 100 == 0:
    #    torch.save(model.state_dict(), f'Schrodingers_Equation/models/model_LBFGS_{iter}.pt')
    return loss


def train(model, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, epochs, lr, summary):
    # Initialize the optimizer
    optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr=lr,
                                  max_iter=epochs,
                                  max_eval=epochs,
                                  history_size=100,
                                  tolerance_grad=0.5 * np.finfo(float).eps,
                                  tolerance_change=0.5 * np.finfo(float).eps,
                                  line_search_fn="strong_wolfe")

    closure_fn = partial(closure, model, optimizer, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, summary)
    optimizer.step(closure_fn)


def main(layer, neurons, initpts, bcpts, colpts, epochs, lr):
    iter = 0
    final_relerr = 0
    final_loss = 0
    set_seed(seed)
    args = {'layer': layer, 'neurons': neurons, 'initpts': initpts, 'bcpts': bcpts, 'colpts': colpts, 'epochs': epochs,
            'lr': lr}
    # args_summary(args)
    summary = SummaryWriter(
        comment='NN' + 'l' + str(layer) + '_n' + str(neurons) + '_i' + str(initpts) + '_b' + str(bcpts) + '_col' + str(
            colpts))
    time_start = time.time()
    model = Wave(layer, neurons).to(device)
    model.apply(init_weights)
    x_ic, t_ic, u_ic = initial_point(initpts, seed)
    l_t_bc, u_t_bc = bc_point(bcpts, seed)
    x_f, t_f, u_f = collocation_point(colpts, seed)
    x_ic = Variable(torch.from_numpy(x_ic.astype(np.float32)), requires_grad=True).to(device)
    t_ic = Variable(torch.from_numpy(t_ic.astype(np.float32)), requires_grad=True).to(device)
    u_ic = Variable(torch.from_numpy(u_ic.astype(np.float32)), requires_grad=True).to(device)
    l_t_bc = Variable(torch.from_numpy(l_t_bc.astype(np.float32)), requires_grad=True).to(device)
    u_t_bc = Variable(torch.from_numpy(u_t_bc.astype(np.float32)), requires_grad=False).to(device)
    x_f = Variable(torch.from_numpy(x_f.astype(np.float32)), requires_grad=True).to(device)
    t_f = Variable(torch.from_numpy(t_f.astype(np.float32)), requires_grad=True).to(device)
    u_f = Variable(torch.from_numpy(u_f.astype(np.float32)), requires_grad=True).to(device)
    train(model, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, epochs, lr, summary)
    summary.add_hparams(args, {'loss': final_loss, 'rel_error': final_relerr * 100})
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    plot(model, summary, device)


if __name__ == "__main__":
    for layer in range(2,9):
        for neurons in range(10,50,10):
            iter = 0
            final_loss = 0
            final_relerr = 0
            main(layer, neurons, 50, 50, 5000, 1000, 1)
