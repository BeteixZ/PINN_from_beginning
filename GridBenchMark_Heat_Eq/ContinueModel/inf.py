import os

from GridBenchMark_Heat_Eq.ContinueModel.functional import setSeed

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from functional import plot_t_error
from model import HeatEqModel

import torch
import matplotlib.pyplot as plt
from functional import args_summary
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--layer', help='number of layers', type=int, default=4)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=50)
parser.add_argument('--AEpoch', help='Number of ADAM epochs', type=int, default=1000)
parser.add_argument('--LEpoch', help='Number of LBFGS epochs', type=int, default=2000)
parser.add_argument('--act', help='Activation function', type=str, default='tanh')
parser.add_argument('--save', help='save models', type=bool, default=True)
parser.add_argument('--record', help='Make Tensorboard record', type=bool, default=True)
parser.add_argument('--seed', help='Random seed', type=int, default=127)
args = parser.parse_args()
args_summary(args)

setSeed(127)

nnPara = [args.layer, args.neurons, args.act]
iterPara = [args.AEpoch, args.LEpoch]

#pts = dataGen(24, 24, 24, True, 42)
model = HeatEqModel(iterPara = iterPara, nnPara= nnPara)
model.loadFCModel('./models/5000-12x12x12_nort.pt')
model.model.eval()


for time in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    colorbar = True
    fig = plot_t_error(model.model, model.device, 3000, time, colorbar)
    fig.savefig("./outputpics/" + 'final_error_pic-' + str(time) + '.png')
    plt.close('all')