import numpy as np
from pyDOE import lhs
from functional import set_seed


def ptsgen(in_size=201, out_size=201, wall_size=441, cy_size=251, coll_size1=22500, coll_size2=7500, seed=42):
    set_seed(seed)
    lb = np.array([0, 0])
    ub = np.array([1.1, 0.41])
    # WALL = [x, y], u=v=0
    wall_up = [0.0, 0.41] + [1.1, 0.0] * lhs(2, wall_size)
    wall_lw = [0.0, 0.00] + [1.1, 0.0] * lhs(2, wall_size)
    # INLET = [x, y, u, v]
    U_max = 1.0
    INLET = [0.0, 0.0] + [0.0, 0.41] * lhs(2, in_size)
    y_INLET = INLET[:, 1:2]
    u_INLET = 4 * U_max * y_INLET * (0.41 - y_INLET) / (0.41 ** 2)  # initial condition
    v_INLET = 0 * y_INLET
    INLET = np.concatenate((INLET, u_INLET, v_INLET), 1)
    # plt.scatter(INLET[:, 1:2], INLET[:, 2:3], marker='o', alpha=0.2, color='red')
    # plt.show()
    # INLET = [x, y], p=0
    OUTLET = [1.1, 0.0] + [0.0, 0.41] * lhs(2, out_size)
    # Cylinder surface
    r = 0.05
    theta = [0.0] + [2 * np.pi] * lhs(1, cy_size)
    x_CYLD = np.multiply(r, np.cos(theta)) + 0.2
    y_CYLD = np.multiply(r, np.sin(theta)) + 0.2
    CYLD = np.concatenate((x_CYLD, y_CYLD), 1)
    WALL = np.concatenate((CYLD, wall_up, wall_lw), 0)
    # Collocation point for equation residual
    XY_c = lb + (ub - lb) * lhs(2, coll_size1)
    XY_c_refine = [0.1, 0.1] + [0.2, 0.2] * lhs(2, coll_size2)
    XY_c = np.concatenate((XY_c, XY_c_refine), 0)
    XY_c = DelCylPT(XY_c, xc=0.2, yc=0.2, r=0.05)
    # XY_c = np.concatenate((XY_c, WALL, CYLD, OUTLET, INLET[:, 0:2]), 0)

    return XY_c[:, 0], XY_c[:, 1], INLET[:, 0], INLET[:, 1], INLET[:, 2],INLET[:, 3], OUTLET[:, 0], OUTLET[:, 1], WALL[:, 0], WALL[:, 1]


def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''
    delete points within cylinder
    '''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst > r, :]
