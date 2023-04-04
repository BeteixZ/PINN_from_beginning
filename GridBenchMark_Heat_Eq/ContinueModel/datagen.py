import numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs
from functional import setSeed


def dataGen(numX, numY, numT, grid=True, randSeed=42):
    lowB = [0,0,0]
    upperB = [1,1,1]

    setSeed(randSeed)

    ptsAll = None
    ptsBD = None
    ptsIC = None

    if grid is True:
        x = np.linspace(lowB[0], upperB[0], numX)
        y = np.linspace(lowB[1], upperB[1], numY)
        t = np.linspace(lowB[2], upperB[2], numT)
        X, Y, T = np.meshgrid(x, y, t)
        X = X.flatten()
        Y = Y.flatten()
        T = T.flatten()
        ptsAll = np.transpose([X, Y, T])

        # IC
        ptsIC = np.array(list((map(lambda x: np.extract(ptsAll[:,2] == 0.0, ptsAll[:,x]), [0,1,2])))).transpose()

        # BC
        ptsBD_x0y = np.array(
            list(map(lambda x: np.extract(np.bitwise_and(ptsAll[:, 0] == 0.0,ptsAll[:, 2] > 0.0), ptsAll[:, x]), [0, 1, 2]))).transpose()
        ptsBD_xy1 = np.array(
            list(map(lambda x: np.extract(np.bitwise_and(ptsAll[:, 1] == 1.0,ptsAll[:, 2] > 0.0), ptsAll[:, x]), [0, 1, 2]))).transpose()
        ptsBD_x1y = np.array(
            list(map(lambda x: np.extract(np.bitwise_and(ptsAll[:, 0] == 1.0,ptsAll[:, 2] > 0.0), ptsAll[:, x]), [0, 1, 2]))).transpose()
        ptsBD_xy0 = np.array(
            list(map(lambda x: np.extract(np.bitwise_and(ptsAll[:, 1] == 0.0,ptsAll[:, 2] > 0.0), ptsAll[:, x]), [0, 1, 2]))).transpose()

        ptsBD = [ptsBD_x0y, ptsBD_xy1, ptsBD_x1y, ptsBD_xy0]

        # CL
        #ptsCL = np.array(list(
        #    map(lambda x: np.extract(np.bitwise_and(ptsAll[:, x] < upperBA[x], ptsAll[:, x] > lowB[x]), ptsAll[:, x]),
        #        [0, 1, 2]))).transpose()

        plt.figure()  # 设置画布大小
        ax = plt.axes(projection='3d')  # 设置三维轴
        ax.scatter3D(ptsBD[0][ :, 0], ptsBD[0][ :, 1], ptsBD[0][ :, 2], c='r')
        ax.scatter3D(ptsBD[1][ :, 0], ptsBD[1][ :, 1], ptsBD[1][ :, 2], c='y')
        ax.scatter3D(ptsBD[2][ :, 0], ptsBD[2][ :, 1], ptsBD[2][ :, 2], c='k')
        ax.scatter3D(ptsBD[3][ :, 0], ptsBD[3][ :, 1], ptsBD[3][ :, 2], c='c')
        ax.scatter3D(ptsIC[:, 0], ptsIC[:, 1], ptsIC[:, 2], c='g')
        ax.scatter3D(ptsAll[:, 0], ptsAll[:, 1], ptsAll[:, 2], c='b')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('T Label')
        plt.show()

        print("Datagen overview:")
        print("Number of collocation points: " + str(ptsAll.size))
        print("Number of boundary points: " + str(ptsBD[0].size*4))
        print("Number of ic points: " + str(ptsIC.size))



    return [ptsAll, ptsBD, ptsIC]
