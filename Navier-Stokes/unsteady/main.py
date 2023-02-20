import os
import shutil

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
    args_summary, postProcess, preprocess
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
parser.add_argument('--layer', help='number of layers', type=int, default=8)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=40)
parser.add_argument('--initpts', help='number of init points pper layer', type=int, default=200)
parser.add_argument('--bcpts', help='number of boundary points', type=int, default=200)
parser.add_argument('--colpts', help='number of collocation points', type=int, default=30000)
parser.add_argument('--epochs', help='number of epochs', type=int, default=30000)
parser.add_argument('--method', help='optimization method', type=str, default='lbfgs')
parser.add_argument('--act', help='activation function', type=str, default='relu')
parser.add_argument('--save', help='save model', type=bool, default=True)


def closure(model, optimizer, x_f, y_f,t_f, x_in, y_in,t_in, u_in, v_in, x_out, y_out,t_out, x_wall, y_wall,t_wall, summary):
    """
    The closure function to use L-BFGS optimization method.
    """
    global iter
    global final_loss
    # global final_relerr
    global count
    optimizer.zero_grad()
    # evaluating the MSE for the PDE
    msef = mse_f(model, x_f, y_f,t_f)
    msein = mse_inlet(model, x_in, y_in,t_in, u_in, v_in)
    mseout = mse_outlet(model, x_out, y_out,t_out)
    msewall = mse_wall(model, x_wall, y_wall,t_wall)
    loss = sum(msef) + 1 * (msein + mseout + msewall)  # 2 here is a parameter??
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


def train(model, x_f, y_f,t_f, x_in, y_in,t_in, u_in, v_in, x_out, y_out,t_out, x_wall, y_wall,t_wall, epochs, summary, epoch):
    # Initialize the optimizer
    global iter

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    print("Start training: ADAM")
    for i in range(5000):
        closure_fn = partial(closure, model, optimizer, x_f, y_f,t_f, x_in, y_in,t_in, u_in, v_in, x_out, y_out,t_out, x_wall, y_wall,t_wall,
                             summary)
        optimizer.step(closure_fn)
        scheduler.step()
    torch.cuda.empty_cache()
    print("Start training: L-BFGS")
    optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr=1,
                                  max_iter=epochs - 5000,
                                  max_eval=epochs - 5000,
                                  history_size=100,
                                  # tolerance_grad=0.01 * np.finfo(float).eps,
                                  tolerance_change=0,
                                  line_search_fn="strong_wolfe")
    closure_fn = partial(closure, model, optimizer, x_f, y_f,t_f, x_in, y_in,t_in, u_in, v_in, x_out, y_out,t_out, x_wall, y_wall,t_wall,
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
    x_f, y_f, t_f, x_in, y_in, t_in, u_in, v_in, x_out, y_out, t_out, x_wall, y_wall, t_wall = ptsgen()  # let's use default first
    x_f = Variable(torch.from_numpy(x_f.astype(np.float32)), requires_grad=True).to(device)
    y_f = Variable(torch.from_numpy(y_f.astype(np.float32)), requires_grad=True).to(device)
    t_f = Variable(torch.from_numpy(t_f.astype(np.float32)), requires_grad=True).to(device)

    x_in = Variable(torch.from_numpy(x_in.astype(np.float32)), requires_grad=True).to(device)
    y_in = Variable(torch.from_numpy(y_in.astype(np.float32)), requires_grad=True).to(device)
    t_in = Variable(torch.from_numpy(t_in.astype(np.float32)), requires_grad=True).to(device)
    u_in = Variable(torch.from_numpy(u_in.astype(np.float32)), requires_grad=False).to(device)
    v_in = Variable(torch.from_numpy(v_in.astype(np.float32)), requires_grad=False).to(device)

    x_out = Variable(torch.from_numpy(x_out.astype(np.float32)), requires_grad=True).to(device)
    y_out = Variable(torch.from_numpy(y_out.astype(np.float32)), requires_grad=True).to(device)
    t_out = Variable(torch.from_numpy(t_out.astype(np.float32)), requires_grad=True).to(device)


    x_wall = Variable(torch.from_numpy(x_wall.astype(np.float32)), requires_grad=True).to(device)
    y_wall = Variable(torch.from_numpy(y_wall.astype(np.float32)), requires_grad=True).to(device)
    t_wall = Variable(torch.from_numpy(t_wall.astype(np.float32)), requires_grad=True).to(device)

    train(model, x_f, y_f,t_f, x_in, y_in,t_in, u_in, v_in, x_out, y_out,t_out, x_wall, y_wall,t_wall, args.epochs, summary,
          args.epochs)
    summary.add_hparams(vars(args), {'loss': final_loss})
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    if args.save:
        torch.save(model.state_dict(), './models/' + 'l' + str(args.layer) + '_n' + str(args.neurons) + '_i' + str(
            args.initpts) + '_b' + str(args.bcpts) + '_col' + str(args.colpts / 1000) + '-' + str(
            args.method) + '-' + str(args.act) + '.pt')
        # Plot the pressure history on the leading point of cylinder

    t_front = np.linspace(0, 0.5, 100)
    x_front = np.zeros_like(t_front)
    x_front.fill(0.15)
    y_front = np.zeros_like(t_front)
    y_front.fill(0.20)
    t_front = t_front.flatten()[:, None]
    x_front = x_front.flatten()[:, None]
    y_front = y_front.flatten()[:, None]

    u_pred, v_pred, p_pred = model.predict(x_front, y_front, t_front)
    plt.plot(t_front, p_pred)
    plt.show()

    # Output u, v, p at each time step
    N_t = 51
    x_star = np.linspace(0, 1.1, 401)
    y_star = np.linspace(0, 0.41, 161)
    x_star, y_star = np.meshgrid(x_star, y_star)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    dst = ((x_star - 0.2) ** 2 + (y_star - 0.2) ** 2) ** 0.5
    x_star = x_star[dst >= 0.05]
    y_star = y_star[dst >= 0.05]
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    shutil.rmtree('./output', ignore_errors=True)
    os.makedirs('./output')
    for i in range(N_t):
        t_star = np.zeros((x_star.size, 1))
        t_star.fill(i * 0.5 / (N_t - 1))

        u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
        field = [x_star, y_star, t_star, u_pred, v_pred, p_pred]
        amp_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5

        postProcess(xmin=0, xmax=1.1, ymin=0, ymax=0.41, field=field, s=2, num=i)

if __name__ == "__main__":

    main()
