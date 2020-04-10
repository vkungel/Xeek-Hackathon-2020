import numpy as np
import scipy.linalg

class FaultFitter:

    def __init__(self, order=1):
        self.order = order

    def get_A_matr(self, x, y):
        if self.order == 1:
            A = np.c_[np.ones(x.shape), x, y]
        elif self.order == 2:
            A = np.c_[np.ones(x.shape), x, y, x*y, x**2, y**2]
        else:
            A = np.zeros(x.shape)
        return A

    def eval_plane(self, x, y, c):
        z = np.dot(self.get_A_matr(x, y), c)
        return z

    def fit_plane(self, x, y, z):
        A = self.get_A_matr(x, y)
        C, _, _, _ = scipy.linalg.lstsq(A, z)  # coefficients
        return C