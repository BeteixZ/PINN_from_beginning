import numpy as np
from pyDOE import lhs
from functional import set_seed


def ptsgen(seed=42):
    set_seed(seed)
    xmax = 2.2
    tmax = 3.   # 0.5
    lb = np.array([0,0,0])
    ub = np.array([xmax, 0.41, tmax])
    x_IC, y_IC, t_IC = CartGrid(xmin=0, xmax=xmax,
                                ymin=0, ymax=0.41,
                                tmin=0, tmax=0,
                                num_x=81, num_y=41, num_t=1)
    IC = np.concatenate((x_IC, y_IC, t_IC), 1)
    IC = DelSrcPT(IC, xc=0.2, yc=0.2, r=0.05)

    x_upb, y_upb, t_upb = CartGrid(xmin=0, xmax=xmax,
                                   ymin=0.41, ymax=0.41,
                                   tmin=0, tmax=tmax,
                                   num_x=81, num_y=1, num_t=41)

    x_lwb, y_lwb, t_lwb = CartGrid(xmin=0, xmax=xmax,
                                   ymin=0, ymax=0,
                                   tmin=0, tmax=tmax,
                                   num_x=81, num_y=1, num_t=41)
    wall_up = np.concatenate((x_upb, y_upb, t_upb), 1)
    wall_lw = np.concatenate((x_lwb, y_lwb, t_lwb), 1)

    U_max = 1.5
    T = tmax * 2  # Period
    x_inb, y_inb, t_inb = CartGrid(xmin=0, xmax=0,
                                   ymin=0, ymax=0.41,
                                   tmin=0, tmax=tmax,
                                   num_x=1, num_y=61, num_t=61)
    u_inb = 4*U_max * y_inb * (0.41 - y_inb)
    #u_inb = 4 * U_max * y_inb * (0.41 - y_inb) / (0.41 ** 2) * (np.sin(2 * 3.1416 * t_inb / T + 3 * 3.1416 / 2) + 1.0)
    v_inb = np.zeros_like(x_inb)
    INB = np.concatenate((x_inb, y_inb, t_inb, u_inb, v_inb), 1)


    x_outb, y_outb, t_outb = CartGrid(xmin=1.1, xmax=1.1,
                                      ymin=0, ymax=0.41,
                                      tmin=0, tmax=tmax,
                                      num_x=1, num_y=81, num_t=41)
    OUTB = np.concatenate((x_outb, y_outb, t_outb), 1)

    # Cylinder surface
    r = 0.05
    x_surf, y_surf, t_surf = GenCirclePT(xc=0.2, yc=0.2, r=r, tmin=0, tmax=tmax, num_r=81, num_t=51)
    HOLE = np.concatenate((x_surf, y_surf, t_surf), 1)

    WALL = np.concatenate((HOLE, wall_up, wall_lw), 0)

    # Collocation point on domain, with refinement near the wall
    XY_c = lb + (ub - lb) * lhs(3, 80000)
    XY_c_refine = [0.0, 0.0, 0.0] + [0.4, 0.4, tmax] * lhs(3, 10000)
    XY_c_lw = [0.0, 0.0, 0.0] + [1.1, 0.02, tmax] * lhs(3, 3000)
    XY_c_up = [0.0, 0.39, 0.0] + [1.1, 0.02, tmax] * lhs(3, 3000)
    XY_c = np.concatenate((XY_c, XY_c_refine, XY_c_lw, XY_c_up), 0)
    XY_c = DelSrcPT(XY_c, xc=0.2, yc=0.2, r=0.05)
    XY_c = np.concatenate((XY_c, WALL, OUTB, INB[:, 0:3]), 0)

    return XY_c[:, 0], XY_c[:, 1], XY_c[:,2], IC[:,0], IC[:,1], IC[:,2], INB[:, 0], INB[:, 1], INB[:, 2],INB[:, 3],INB[:,4], OUTB[:, 0], OUTB[:, 1],OUTB[:,2], WALL[:, 0], WALL[:, 1],WALL[:,2]


def CartGrid(xmin, xmax, ymin, ymax, tmin, tmax, num_x, num_y, num_t):
    # num_x, num_y: number per edge
    # num_t: number time step

    x = np.linspace(xmin, xmax, num=num_x)
    y = np.linspace(ymin, ymax, num=num_y)
    xx, yy = np.meshgrid(x, y)
    t = np.linspace(tmin, tmax, num=num_t)
    xxx, yyy, ttt = np.meshgrid(x, y, t)
    xxx = xxx.flatten()[:, None]
    yyy = yyy.flatten()[:, None]
    ttt = ttt.flatten()[:, None]
    return xxx, yyy, ttt


def GenCirclePT(xc, yc, r, tmin, tmax, num_r, num_t):
    # Generate collocation points at cylinder uniformly
    theta = np.linspace(0.0, np.pi*2.0, num_r)
    x = np.multiply(r, np.cos(theta)) + xc
    y = np.multiply(r, np.sin(theta)) + yc
    t = np.linspace(tmin, tmax, num_t)
    xx, tt = np.meshgrid(x, t)
    yy, _ = np.meshgrid(y, t)
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    tt = tt.flatten()[:, None]

    return xx, yy, tt


def DelSrcPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    # Delete collocation point within cylinder
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]