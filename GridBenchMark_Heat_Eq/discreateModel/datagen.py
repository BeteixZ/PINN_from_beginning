import numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs
from functional import setSeed


def dataGen(numX, numY, grid, randSeed=42):
    lowB = [0,0,0]
    upperB = [1,1,0.5]

    setSeed(randSeed)

    if grid is True:
        x = np.linspace(lowB[0], upperB[0], numX)
        y = np.linspace(lowB[1], upperB[1], numY)
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        ptsAll = np.transpose([X, Y])


        # BC
        ptsBD_x0y = np.array(
            list(map(lambda x: np.extract(ptsAll[:, 0] == 0.0, ptsAll[:, x]), [0, 1]))).transpose()
        ptsBD_xy1 = np.array(
            list(map(lambda x: np.extract(ptsAll[:, 1] == 1.0, ptsAll[:, x]), [0, 1]))).transpose()
        ptsBD_x1y = np.array(
            list(map(lambda x: np.extract(ptsAll[:, 0] == 1.0, ptsAll[:, x]), [0, 1]))).transpose()
        ptsBD_xy0 = np.array(
            list(map(lambda x: np.extract(ptsAll[:, 1] == 0.0, ptsAll[:, x]), [0, 1]))).transpose()

        ptsBD = [ptsBD_x0y, ptsBD_xy1, ptsBD_x1y, ptsBD_xy0]

        # CL
        ptsCL = np.array(list(
            map(lambda x: np.extract(np.bitwise_and(ptsAll[:, x] <= upperB[x], ptsAll[:, x] >= lowB[x]), ptsAll[:, x]),
                [0, 1]))).transpose()

        #plt.figure()
        #ax = plt.axes()
        #ax.scatter(ptsCL[: , 0], ptsCL[: , 1])
        #ax.scatter(ptsBD[0][ :, 0], ptsBD[0][ :, 1])
        #ax.scatter(ptsBD[1][ :, 0], ptsBD[1][: , 1])
        #ax.scatter(ptsBD[2][ :, 0], ptsBD[2][: , 1])
        #ax.scatter(ptsBD[3][ :, 0], ptsBD[3][ :, 1])
        #plt.show()

        return [ptsCL, ptsBD]

# UNIT TEST PASS