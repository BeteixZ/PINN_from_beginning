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
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from torch.autograd import Variable
import time
import argparse

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from functional import set_seed, init_weights, args_summary, plot, plot_with_points, make_gif, plot_slice, plot_error
from model import Wave, mse_f, mse_0, mse_b, rel_error, mse_data
from datagen import initial_point, bc_point, collocation_point, mesh_point

iter = 0
count = 1
seed = 42
final_loss = 0
final_relerr = 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--layer', help='number of layers', type=int, default=6)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=50)
parser.add_argument('--initpts', help='number of init points pper layer', type=int, default=50)
parser.add_argument('--bcpts', help='number of boundary points', type=int, default=50)
parser.add_argument('--colpts', help='number of collocation points', type=int, default=9000)
parser.add_argument('--epochs', help='number of epochs', type=int, default=1000)
parser.add_argument('--lr', help='learning rate', type=float, default=1)
parser.add_argument('--method', help='optimization method', type=str, default='lbfgs')
parser.add_argument('--act', help='activation function', type=str, default='sigmoid')


def closure(model, optimizer, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, summary):
    """
    The closure function to use L-BFGS optimization method.
    """
    global iter
    global final_loss
    global final_relerr
    global count
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
    if iter % 1 == 0:
        print(f"[Iter: {iter}] loss: {loss.item()}, msef:{msef}, mse0:{mse0}, mseb:{mseb}, relerr:{relerror}")
    slice = np.concatenate((np.arange(0,10,1), np.arange(10,100,10), np.arange(100,1100,100)))
    if iter in slice:
        fig = plot(model, device, iter)
        fig.savefig("./outputpics/" + 'pic-'+str(count) + '.png')
        plt.close('all')
        count+=1
    return loss


def train(model, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, epochs, lr, method, summary):
    # Initialize the optimizer

    if method == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(),
                                    lr=lr,
                                    max_iter=epochs,
                                    max_eval=epochs,
                                    history_size=100,
                                    #tolerance_grad=0.01 * np.finfo(float).eps,
                                    tolerance_change=0,
                                    line_search_fn="strong_wolfe")
        closure_fn = partial(closure, model, optimizer, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, summary)
        optimizer.step(closure_fn)


    if method == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        schedule2 = ExponentialLR(optimizer, gamma= 0.9995)
        for i in range(epochs):
            closure_fn = partial(closure, model, optimizer, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, summary)
            optimizer.step(closure_fn)
            schedule2.step()

    if method == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        schedule2 =  ExponentialLR(optimizer, gamma= 0.999)
        for i in range(epochs):
            closure_fn = partial(closure, model, optimizer, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, summary)
            optimizer.step(closure_fn)
            schedule2.step()

    if method == 'ada':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.02)
        schedule2 =  ExponentialLR(optimizer, gamma= 0.999)
        for i in range(epochs):
            closure_fn = partial(closure, model, optimizer, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, summary)
            optimizer.step(closure_fn)
            schedule2.step()

    if method == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.002)
        schedule2 =  ExponentialLR(optimizer, gamma= 0.995)
        for i in range(epochs):
            closure_fn = partial(closure, model, optimizer, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, summary)
            optimizer.step(closure_fn)
            schedule2.step()


def main():
    set_seed(seed)
    args = parser.parse_args()
    args_summary(args)
    summary = SummaryWriter(comment= 'l'+str(args.layer)+'_n'+str(args.neurons)+'_i' + str(args.initpts) + '_b' + str(args.bcpts)+'_col'+str(args.colpts/1000)+'-'+str(args.method)+'-'+str(args.act))
    time_start = time.time()
    model = Wave(args.layer, args.neurons, args.act).to(device)
    model.apply(init_weights)
    x_ic, t_ic, u_ic = initial_point(args.initpts, seed)
    l_t_bc, u_t_bc = bc_point(args.bcpts, seed)
    x_f, t_f, u_f = collocation_point(args.colpts, seed)
    x_ic = Variable(torch.from_numpy(x_ic.astype(np.float32)), requires_grad=False).to(device)
    t_ic = Variable(torch.from_numpy(t_ic.astype(np.float32)), requires_grad=False).to(device)
    u_ic = Variable(torch.from_numpy(u_ic.astype(np.float32)), requires_grad=False).to(device)
    l_t_bc = Variable(torch.from_numpy(l_t_bc.astype(np.float32)), requires_grad=True).to(device)
    u_t_bc = Variable(torch.from_numpy(u_t_bc.astype(np.float32)), requires_grad=False).to(device)
    x_f = Variable(torch.from_numpy(x_f.astype(np.float32)), requires_grad=True).to(device)
    t_f = Variable(torch.from_numpy(t_f.astype(np.float32)), requires_grad=True).to(device)
    u_f = Variable(torch.from_numpy(u_f.astype(np.float32)), requires_grad=True).to(device)
    train(model, x_f, t_f, u_f, x_ic, t_ic, u_ic, l_t_bc, u_t_bc, args.epochs, args.lr, args.method, summary)
    summary.add_hparams(vars(args), {'loss':final_loss, 'rel_error':final_relerr*100})
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    make_gif()
    plot_slice(model, summary, device)
    plot_error(model, summary, device)
    plot_with_points(model, t_ic, x_ic, l_t_bc, u_t_bc, summary, device)


if __name__ == "__main__":
    main()
