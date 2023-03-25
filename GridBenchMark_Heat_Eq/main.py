import argparse
from datagen import dataGen
from functional import setSeed
from GridBenchMark_Heat_Eq.model import HeatEqModel

parser = argparse.ArgumentParser()
parser.add_argument('--layer', help='number of layers', type=int, default=5)
parser.add_argument('--neurons', help='number of neurons per layer', type=int, default=50)
parser.add_argument('--numIn', help='number of init points pper layer', type=int, default=500)
parser.add_argument('--numOut', help='number of boundary points', type=int, default=500)
parser.add_argument('--numCL', help='number of collocation points', type=int, default=20000)
parser.add_argument('--numOB', help='number of obstacle points', type=int, default=500)
parser.add_argument('--numIC', help='number of initial points', type=int, default=500)
parser.add_argument('--AEpoch', help='Number of ADAM epochs', type=int, default=10000)
parser.add_argument('--LEpoch', help='Number of LBFGS epochs', type=int, default=10000)
parser.add_argument('--act', help='Activation function', type=str, default='tanh')
parser.add_argument('--save', help='save models', type=bool, default=True)
parser.add_argument('--record', help='Make Tensorboard record', type=bool, default=True)
parser.add_argument('--seed', help='Random seed', type=int, default=42)


def main():
    args = parser.parse_args()
    print(args)

    setSeed(args.seed)

    lowerBound = [0, 0, 0]
    upperBound = [2.2, 0.41, 0.5]
    cyldCoord = [0.2, 0.2]
    cyldRadius = 0.05

    bound = [lowerBound, upperBound]
    nnPara = [args.layer, args.neurons, args.act]
    iterPara = [args.AEpoch, args.LEpoch]

    pts = dataGen(32, 32, 32 ,42)
    model = HeatEqModel(pts, bound, nnPara, iterPara, args.save, args.record, args.seed)


    # models.loadFCModel("./models/dortmund-2d-2-unsteay-models.pt")
    model.train()
    model.inference()


if __name__ == "__main__":
    main()
