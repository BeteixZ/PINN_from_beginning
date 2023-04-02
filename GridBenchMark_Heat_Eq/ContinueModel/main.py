import argparse

from matplotlib import pyplot as plt

from datagen import dataGen
from functional import setSeed
from GridBenchMark_Heat_Eq.ContinueModel.model import HeatEqModel

parser = argparse.ArgumentParser()
parser.add_argument('--layer', help='number of layers', type=int, default=4)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=50)
parser.add_argument('--AEpoch', help='Number of ADAM epochs', type=int, default=2500)
parser.add_argument('--LEpoch', help='Number of LBFGS epochs', type=int, default=2500)
parser.add_argument('--act', help='Activation function', type=str, default='tanh')
parser.add_argument('--save', help='save models', type=bool, default=True)
parser.add_argument('--record', help='Make Tensorboard record', type=bool, default=True)
parser.add_argument('--seed', help='Random seed', type=int, default=127)


def main():
    args = parser.parse_args()
    print(args)

    setSeed(args.seed)

    lowerBound = [0, 0, 0]
    upperBound = [2.2, 0.41, 0.5]


    bound = [lowerBound, upperBound]
    nnPara = [args.layer, args.neurons, args.act]
    iterPara = [args.AEpoch, args.LEpoch]

    pts = dataGen(24, 24, 24, True, 42)
    model = HeatEqModel(nnPara, iterPara,pts, bound, "ADAM", args.save, args.record, "12x12x12_nort", args.seed)


    # models.loadFCModel("./models/dortmund-2d-2-unsteay-models.pt")
    model.train()
    model.inference()


    #plt.figure()  # 设置画布大小
    #ax = plt.axes(projection='3d')  # 设置三维轴
    #ax.scatter3D(model.ptsCl[:, 0], model.ptsCl[:, 1], model.ptsCl[:, 2], c='b')
    #plt.show()


if __name__ == "__main__":
    main()
