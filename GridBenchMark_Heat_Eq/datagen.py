import numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs
from functional import setSeed


def dataGen(numX, numY, numT, grid, randSeed=42):
    lowB = [0,0,0]
    upperB = [1,1,0.5]

    setSeed(randSeed)

    if grid is True:
        x = np.linspace(lowB[0], upperB[0], numX)
        y = np.linspace(lowB[1], upperB[1], numY)
        t = np.linspace(lowB[2], upperB[2], numY)
        X, Y, T = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        T = T.flatten()
        ptsAll = np.transpose([X, Y, T])

        # IC
        ptsIC = np.array(list((map(lambda x: np.extract(ptsAll[:,2] == 0.0, ptsAll[:,x]), [0,1,2])))).transpose()

        # BC
        ptsBD_x0y = np.array(
            list(map(lambda x: np.extract(ptsAll[:, 0] == 0.0, ptsAll[:, x]), [0, 1, 2]))).transpose()
        ptsBD_xy1 = np.array(
            list(map(lambda x: np.extract(ptsAll[:, 1] == 1.0, ptsAll[:, x]), [0, 1, 2]))).transpose()
        ptsBD_x1y = np.array(
            list(map(lambda x: np.extract(ptsAll[:, 0] == 1.0, ptsAll[:, x]), [0, 1, 2]))).transpose()
        ptsBD_xy0 = np.array(
            list(map(lambda x: np.extract(ptsAll[:, 1] == 0.0, ptsAll[:, x]), [0, 1, 2]))).transpose()

        ptsBD = np.concatenate((ptsBD_x0y, ptsBD_xy1, ptsBD_x1y, ptsBD_xy0))

        # CL
        ptsCL = np.array(list(
            map(lambda x: np.extract(np.bitwise_and(ptsAll[:, x] <= upperB[x], ptsAll[:, x] >= lowB[x]), ptsAll[:, x]),
                [0, 1, 2]))).transpose()

        plt.figure()  # 设置画布大小
        ax = plt.axes(projection='3d')  # 设置三维轴
        ax.scatter3D(ptsBD[0, :, 0], ptsBD[0, :, 1], ptsBD[0, :, 2])
        ax.scatter3D(ptsBD[1, :, 0], ptsBD[1, :, 1], ptsBD[1, :, 2])
        ax.scatter3D(ptsBD[2, :, 0], ptsBD[2, :, 1], ptsBD[2, :, 2])
        ax.scatter3D(ptsBD[3, :, 0], ptsBD[3, :, 1], ptsBD[3, :, 2])

    return [ptsCL, ptsIC, ptsBD]
