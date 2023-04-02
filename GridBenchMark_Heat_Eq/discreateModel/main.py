import argparse

import torch
from matplotlib import pyplot as plt

from datagen import dataGen
from functional import setSeed
from model import HeatEqModel

parser = argparse.ArgumentParser()
parser.add_argument('--layer', help='number of layers', type=int, default=4)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=50)
parser.add_argument('--AEpoch', help='Number of ADAM epochs', type=int, default=2)
parser.add_argument('--LEpoch', help='Number of LBFGS epochs', type=int, default=1)
parser.add_argument('--save', help='save models', type=bool, default=False)
parser.add_argument('--record', help='Make Tensorboard record', type=bool, default=False)
parser.add_argument('--seed', help='Random seed', type=int, default=42)


def main():
    args = parser.parse_args()
    print(args)

    setSeed(args.seed)

    lowerBound = [0, 0]
    upperBound = [1, 1]

    bound = [lowerBound, upperBound]
    nnPara = [args.layer, args.neurons]
    iterPara = [args.AEpoch, args.LEpoch]
    tPara = [0, 0.9]
    q = 10

    pts = dataGen(24, 24, True, 42)
    model = HeatEqModel(nnPara, iterPara, q, pts, bound, tPara, args.save, args.record, "q=10, 0.9", args.seed)

    # model.loadFCModel("./models/5000-q=10, 0.1.pt")
    torch.compile(model)
    model.train()


    # model.inference()


if __name__ == "__main__":
    main()
