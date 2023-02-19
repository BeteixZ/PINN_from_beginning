"""
This file contains the main code for PINN including adjustable
1. Neural Layer
2. Neurons
3. Points (Init, Bc and Collo.)
4. Epochs
5. LR

Every run writes to TensorBoard file.、
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from torch.autograd import Variable
import time
import argparse

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from functional import set_seed, init_weights, \
    args_summary, plot_t, postProcess, preprocess
from model import Model, mse_f, mse_inlet, mse_outlet, mse_wall, uv
from datagen import ptsgen



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

iter = 0
count = 1
seed = 99
final_loss = 0
final_relerr = 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--layer', help='number of layers', type=int, default=7)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=40)
parser.add_argument('--initpts', help='number of init points pper layer', type=int, default=200)
parser.add_argument('--bcpts', help='number of boundary points', type=int, default=200)
parser.add_argument('--colpts', help='number of collocation points', type=int, default=30000)
parser.add_argument('--epochs', help='number of epochs', type=int, default=20000)
parser.add_argument('--method', help='optimization method', type=str, default='lbfgs')
parser.add_argument('--act', help='activation function', type=str, default='mish')
parser.add_argument('--save', help='save model', type=bool, default=True)


def closure(model, optimizer, x_f, y_f, x_in, y_in, u_in, v_in, x_out, y_out, x_wall, y_wall, summary):
    """
    The closure function to use L-BFGS optimization method.
    """
    global iter
    global final_loss
    # global final_relerr
    global count
    optimizer.zero_grad()
    # evaluating the MSE for the PDE
    msef = mse_f(model, x_f, y_f)
    msein = mse_inlet(model, x_in, y_in, u_in, v_in)
    mseout = mse_outlet(model, x_out, y_out)
    msewall = mse_wall(model, x_wall, y_wall)
    loss = sum(msef) + 2 * (msein + mseout + msewall)  # 2 here is a parameter??
    # pt_x, pt_y, pt_t, pt_u = mesh_point()
    # pt_x = Variable(torch.from_numpy(pt_x).float(), requires_grad=False).to(device)
    # pt_y = Variable(torch.from_numpy(pt_y).float(), requires_grad=False).to(device)
    # pt_t = Variable(torch.from_numpy(pt_t).float(), requires_grad=False).to(device)
    # relerror = rel_error(model, pt_x, pt_y, pt_t, pt_u)
    summary.add_scalar('Loss', loss, iter)
    summary.add_scalars('MSE', {'MSE_f': sum(msef), 'MSE_0': msein, 'MSE_bc': mseout + msewall}, iter)
    # summary.add_scalar('MSE_relerror', relerror, iter)
    loss.backward(retain_graph=True)
    iter += 1
    final_loss = loss
    # final_relerr = relerror
    if iter % 10 == 0:
        print("Iter: {}, loss: {:.4f}, msef: {}, mse0: {:.4f}, mseb:{:.4f}".format(iter,loss.item(),msef,msein,mseout+msewall))
        # print(f"[Iter: {iter}] loss: {loss.item()}, msef:{msef}, mse0:{msein}, mseb:{mseout + msewall}")
    # slice = np.concatenate((np.arange(0, 10, 1), np.arange(10, 100, 10), np.arange(100, 1100, 100)))
    # if iter in slice:
    #    fig = plot(model, device, iter)
    #    fig.savefig("./outputpics/" + 'pic-' + str(count) + '.png')
    #    plt.close('all')
    #    count += 1
    # if iter == epoch + 200:
    #    for time in [0, 0.25, 0.5, 0.75, 1.0]:
    #        fig = plot_t(model, device, iter, time, True)
    #        fig.savefig("./outputpics/" + 'final_pic-' + str(time) + '.pdf')
    #        plt.close('all')
    return loss


def train(model, x_f, y_f, x_in, y_in, u_in, v_in, x_out, y_out, x_wall, y_wall, epochs, summary, epoch):
    # Initialize the optimizer
    global iter
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Start training: ADAM")
    for i in range(3000):
        closure_fn = partial(closure, model, optimizer, x_f, y_f, x_in, y_in, u_in, v_in, x_out, y_out, x_wall, y_wall,
                             summary)
        optimizer.step(closure_fn)
    torch.cuda.empty_cache()
    print("Start training: L-BFGS")
    optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr=1,
                                  max_iter=epochs - 3000,
                                  max_eval=epochs - 3000,
                                  history_size=50,
                                  # tolerance_grad=0.01 * np.finfo(float).eps,
                                  tolerance_change=0,
                                  line_search_fn="strong_wolfe")
    closure_fn = partial(closure, model, optimizer, x_f, y_f, x_in, y_in, u_in, v_in, x_out, y_out, x_wall, y_wall,
                         summary)
    optimizer.step(closure_fn)


def main():
    set_seed(seed)
    args = parser.parse_args()
    args_summary(args)
    summary = SummaryWriter(
        comment='l' + str(args.layer) + '_n' + str(args.neurons) + '_i' + str(args.initpts) + '_b' + str(
            args.bcpts) + '_col' + str(args.colpts / 1000) + '-' + str(args.method) + '-' + str(args.act))
    time_start = time.time()
    model = Model(args.layer, args.neurons, args.act).to(device)
    model.apply(init_weights)
    x_f, y_f, x_in, y_in, u_in, v_in, x_out, y_out, x_wall, y_wall = ptsgen()  # let's use default first
    x_f = Variable(torch.from_numpy(x_f.astype(np.float32)), requires_grad=True).to(device)
    y_f = Variable(torch.from_numpy(y_f.astype(np.float32)), requires_grad=True).to(device)
    x_in = Variable(torch.from_numpy(x_in.astype(np.float32)), requires_grad=True).to(device)
    y_in = Variable(torch.from_numpy(y_in.astype(np.float32)), requires_grad=True).to(device)
    u_in = Variable(torch.from_numpy(u_in.astype(np.float32)), requires_grad=False).to(device)
    v_in = Variable(torch.from_numpy(v_in.astype(np.float32)), requires_grad=False).to(device)
    x_out = Variable(torch.from_numpy(x_out.astype(np.float32)), requires_grad=True).to(device)
    y_out = Variable(torch.from_numpy(y_out.astype(np.float32)), requires_grad=True).to(device)
    x_wall = Variable(torch.from_numpy(x_wall.astype(np.float32)), requires_grad=True).to(device)
    y_wall = Variable(torch.from_numpy(y_wall.astype(np.float32)), requires_grad=True).to(device)
    train(model, x_f, y_f, x_in, y_in, u_in, v_in, x_out, y_out, x_wall, y_wall, args.epochs, summary,
          args.epochs)
    summary.add_hparams(vars(args), {'loss': final_loss})
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    if args.save:
        torch.save(model.state_dict(), './models/' + 'l' + str(args.layer) + '_n' + str(args.neurons) + '_i' + str(
            args.initpts) + '_b' + str(args.bcpts) + '_col' + str(args.colpts / 1000) + '-' + str(
            args.method) + '-' + str(args.act) + '.pt')
    # make_gif()
    # plot_slice(model, summary, device)
    # plot_error(model, summary, device)
    # plot_with_points(model, t_ic, x_ic, l_t_bc, u_t_bc, summary, device)
    [x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT] = preprocess(dir='../FluentReferenceMu002/FluentSol.mat')
    field_FLUENT = [x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT]

    x_PINN = np.linspace(0, 1.1, 251)
    y_PINN = np.linspace(0, 0.41, 101)
    x_PINN, y_PINN = np.meshgrid(x_PINN, y_PINN)
    x_PINN = x_PINN.flatten()[:, None]
    y_PINN = y_PINN.flatten()[:, None]
    dst = ((x_PINN - 0.2) ** 2 + (y_PINN - 0.2) ** 2) ** 0.5
    x_PINN = x_PINN[dst >= 0.05]
    y_PINN = y_PINN[dst >= 0.05]
    x_PINN = x_PINN.flatten()[:, None]
    y_PINN = y_PINN.flatten()[:, None]
    x_PINN = Variable(torch.from_numpy(x_PINN.astype(np.float32)), requires_grad=True).to(device)
    y_PINN = Variable(torch.from_numpy(y_PINN.astype(np.float32)), requires_grad=True).to(device)

    u_PINN, v_PINN, p_PINN, _, _, _ = uv(model, x_PINN[:, 0], y_PINN[:, 0])
    x_PINN = x_PINN.data.cpu().numpy()
    y_PINN = y_PINN.data.cpu().numpy()
    u_PINN = u_PINN.data.cpu().numpy()
    u_PINN = np.reshape(u_PINN, (u_PINN.size, 1))
    v_PINN = v_PINN.data.cpu().numpy()
    v_PINN = np.reshape(v_PINN, (u_PINN.size, 1))
    p_PINN = p_PINN.data.cpu().numpy()
    field_MIXED = [x_PINN, y_PINN, u_PINN, v_PINN, p_PINN]


    if args.save:
        torch.save(model.state_dict(), './models/'+'model.pt')

        # Plot the comparison of u, v, p
        postProcess(xmin=0, xmax=1.1, ymin=0, ymax=0.41, field_FLUENT=field_FLUENT, field_MIXED=field_MIXED, s=3,
                    alpha=0.5)

if __name__ == "__main__":

    main()
