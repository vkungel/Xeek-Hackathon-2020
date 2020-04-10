import numpy as np
from lib.fitting import FaultFitter
import math
import pandas as pd

def get_strike_dip(fault):
    """Function to compute strike and dip of the fault"""

    if fault.C is None:
        raise RuntimeError('Unable to compute parameters, fault shall be fitted. Please call Fault::fit() first.')

    strike = np.arctan2(fault.C[2], fault.C[1])
    dip = np.arctan(math.sqrt(fault.C[1] ** 2 + fault.C[2] ** 2))

    if strike < 0.:
        strike += 2.*np.pi

    return strike, dip


def get_rotated_points(strike, dip, data):
    """Function which implements rotation of the points.

    3D points of the fault are rotated by strike and dip angles
    in a way that the resulting fault plane is parallel to XY-plane

    Parameters
    ----------
    strike: float
        Strike of the fault
    dip: float
        Dip of the fault
    data: numpy.ndarray
        array of fault points [npoints, 3]

    Returns
    -------
    data_rot: numpy.ndarray
        array of rotated fault points [npoints, 3]
    """

    r_y = np.matrix([[math.cos(dip), 0, math.sin(dip)],
                     [0, 1, 0],
                     [-math.sin(dip), 0, math.cos(dip)]])

    r_z = np.matrix([[math.cos(-strike), -math.sin(-strike), 0],
                     [math.sin(-strike), math.cos(-strike), 0],
                     [0, 0, 1]])

    r_x = np.matrix([[1, 0, 0],
                     [0, math.cos(-dip), -math.sin(-dip)],
                     [0, math.sin(-dip), math.cos(-dip)]])

    data_rot = np.zeros(data.shape)
    for i in range(len(data)):
        data_rot[i:i + 1] = (r_y * r_z * data[i:i + 1].T).T

    return data_rot


def get_area(points_rotated):
    """Comute an area of the fault.

    Area of the fault is computed using convex hull,
    which is implemented as a part of numpy module
    (scipy.spatial.ConvexHull). Input points are projected onto
    XY plane, and used to build 2-D convex hull.
    The area of the convex hull would represent an area of the fault.


    Parameters
    ----------
    points_rotated: numpy.ndarray
        array of rotated fault points.

    Returns
    -------
    area: float
        area of the fault
    """

    # area with monte carlo
    from scipy.spatial import ConvexHull
    data_hull = points_rotated[:, :2].copy()

    # hull_minx = np.min(data_hull[:, 0])
    # hull_miny = np.min(data_hull[:, 1])
    # print(hull_minx, hull_miny)
    #
    # data_hull[:, 0] -= hull_minx
    # data_hull[:, 1] -= hull_miny

    hull = ConvexHull(data_hull)

    area = hull.volume # use volume attibute to get an area of 2D hull

    return area

def curv_x(x,y,C):
    c1 = C[1]
    c2 = C[2]
    c3 = C[3]
    c4 = C[4]
    c5 = C[5]
    return  -1.*(-((c1 + 2 * c4 * x + c3 * y) * (
                4 * c4 * (c1 + 2 * c4 * x + c3 * y) + 2 * c3 * (c2 + c3 * x + 2 * c5 * y))) / (
                    2 * (1 + (c1 + 2 * c4 * x + c3 * y) ** 2 + (c2 + c3 * x + 2 * c5 * y) ** 2) ** (3 / 2)) + (
                    2 * c4) / np.sqrt(1 + (c1 + 2 * c4 * x + c3 * y) ** 2 + (c2 + c3 * x + 2 * c5 * y) ** 2))

def curv_y(x,y,C):
    c1 = C[1]
    c2 = C[2]
    c3 = C[3]
    c4 = C[4]
    c5 = C[5]
    return  -1.*(- (
                    (c2 + c3 * x + 2 * c5 * y) * (
                        2 * c3 * (c1 + 2 * c4 * x + c3 * y) + 4 * c5 * (c2 + c3 * x + 2 * c5 * y))) / (
                    2 * (1 + (c1 + 2 * c4 * x + c3 * y) ** 2 + (c2 + c3 * x + 2 * c5 * y) ** 2) ** (3 / 2)) + (
                    2 * c5) / np.sqrt(1 + (c1 + 2 * c4 * x + c3 * y) ** 2 + (c2 + c3 * x + 2 * c5 * y) ** 2))


def get_mean_curvature(x, y, C):

    return 0.5* (curv_x(x,y,C)+curv_y(x,y,C))


def generate_table(faults):
    df = pd.concat([fault.get_pd_row() for fault in faults], ignore_index = True)
    return df


def process_faults(faults):
    for fault in faults:
        fault.fit()
        fault.compute_params()
    return faults

def center_data(data):
    m_x = np.mean(data[:,0])
    m_y = np.mean(data[:,1])
    m_z = np.mean(data[:,2])
    data[:, 0] -= m_x
    data[:, 1] -= m_y
    data[:, 2] -= m_z
    return data


class Fault:

    def __init__(self, idx, data, order=1):
        # label of the fault
        self.idx = idx
        # nPoints x 3 matrix
        self.data = data
        # vector of x coordinates of the fault points
        self.x = self.data[:,0]
        # vector of y coordinates of the fault points
        self.y = self.data[:,1]
        # vector of z coordinates of the fault points
        self.z = self.data[:,2]
        # plane coefficients
        self.C = None
        self.C_sq = None
        # fitter object
        self.fitter = None
        self.order = order

        # self.data = center_data(self.data)


    def fit(self):
        self.fitter = FaultFitter(self.order)
        self.C = self.fitter.fit_plane(self.x, self.y, self.z)
        self.z_fit = self.fitter.eval_plane(self.x, self.y, self.C)

        fitter_sq = FaultFitter(order=2)
        self.C_sq = fitter_sq.fit_plane(self.x, self.y, self.z)
        self.z_fit_sq = fitter_sq.eval_plane(self.x, self.y, self.C_sq)

    def compute_params(self):
        # self.dip = get_dip(self)
        # self.strike = get_strike(self)

        self.strike, self.dip = get_strike_dip(self)

        self.data_rot = get_rotated_points(self.strike, self.dip, self.data)

        # max height + lenght
        height_max = np.max(self.data_rot[:, 0])
        height_min = np.min(self.data_rot[:, 0])
        self.height = np.abs(height_max - height_min)

        lenght_max = np.max(self.data_rot[:, 1])
        lenght_min = np.min(self.data_rot[:, 1])
        self.length = np.abs(lenght_max - lenght_min)

        self.depth_min = np.max(self.data_rot[:, 2])
        self.depth_max = np.min(self.data_rot[:, 2])

        self.dev = np.max(self.data_rot[:, 2]) - np.min(self.data_rot[:, 2])
        self.std = np.std(self.data_rot[:, 2])

        self.area = get_area(self.data_rot)

        m_x = np.mean(self.data_rot[:, 0])
        m_y = np.mean(self.data_rot[:, 1])


        fitter_sq = FaultFitter(order=2)
        C_sq_rot = fitter_sq.fit_plane(self.data_rot[:,0], self.data_rot[:,1], self.data_rot[:,2])
        # self.z_fit_sq = fitter_sq.eval_plane(self.x, self.y, self.C_sq)

        self.curv = get_mean_curvature(m_x, m_y, C_sq_rot)
        self.curv_x = curv_x(m_x, m_y, C_sq_rot)
        self.curv_y = curv_y(m_x, m_y, C_sq_rot)

        curvs = []
        # curv.shape
        for i in range(self.data_rot.shape[0]):
            curvs.append(get_mean_curvature(self.data_rot[i, 0], self.data_rot[i, 1], C_sq_rot))

        self.curv_mean = np.mean(curvs)
        self.curv_min = np.min(curvs)
        self.curv_max = np.max(curvs)

    def get_pd_row(self):
        df = pd.DataFrame({
            'strike': self.strike,
            'dip': self.dip,
            'depth_min': self.depth_min,
            'depth_max': self.depth_max,
            'height': self.height,
            'length': self.length,
            'area': self.area,
            'curv': self.curv,
            'curv_mean': self.curv_mean,
            'curv_min': self.curv_min,
            'curv_max': self.curv_max,
            'curv_x': self.curv_x,
            'curv_y': self.curv_y,
        }, index=[0])
        return df