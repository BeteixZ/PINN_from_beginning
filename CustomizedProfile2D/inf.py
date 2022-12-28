import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from functional import plot_t
from model import Wave

import torch
import matplotlib.pyplot as plt
from functional import args_summary
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--layer', help='number of layers', type=int, default=6)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=60)
parser.add_argument('--initpts', help='number of init points pper layer', type=int, default=200)
parser.add_argument('--bcpts', help='number of boundary points', type=int, default=200)
parser.add_argument('--colpts', help='number of collocation points', type=int, default=17000)
parser.add_argument('--epochs', help='number of epochs', type=int, default=1200)
parser.add_argument('--lr', help='learning rate', type=float, default=1)
parser.add_argument('--method', help='optimization method', type=str, default='lbfgs')
parser.add_argument('--act', help='activation function', type=str, default='mish')
parser.add_argument('--save', help='save model', type=bool, default=True)
args = parser.parse_args()
args_summary(args)


model = Wave(args.layer, args.neurons, args.act).to(device)
model.load_state_dict(torch.load('./models/l6_n60_i200_b200_col17.0-lbfgs-mish.pt'))
model.eval()

iter = 1200
for time in [0, 0.25, 0.5, 0.75, 1.0]:
    colorbar = False
    if time == 1.0:
        colorbar = True
    else:
        colorbar = False

    fig = plot_t(model, device, iter, time, colorbar)
    fig.savefig("./outputpics/" + 'final_pic-' + str(time) + '.png')
    plt.close('all')