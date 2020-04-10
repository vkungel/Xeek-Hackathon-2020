import numpy as np
from sklearn.cluster import DBSCAN #, OPTICS

def get_labels_slice(data_):
    labels = DBSCAN(min_samples=2,
                eps=1.1,
               ).fit_predict(data_)
    return labels

def get_intersection(data_,
                     min_samples=21,
                     eps=3):

    labels=DBSCAN(min_samples=min_samples,
                  eps=eps
                  ).fit_predict(data_)

    return labels

def get_fault_clusters(data_,
                       min_samples=22,
                       eps=2):

    data_labels = DBSCAN(min_samples=min_samples,
                         eps=eps,
                         ).fit_predict(data_)

    return data_labels


def get_intersection_labels(data,
                            min_samples=12,
                            eps=2):

    labels_2d = np.zeros((data.shape[0]))
    z_min = np.min(data[:, 2])
    z_max = np.max(data[:, 2])
    # iterate along z-axis, and label z slices
    for z in range(z_min, z_max + 1):
        mask = data[:, 2] == z
        if np.any(mask):
            dataAll_slice = data[mask]
            labels_2d[mask] = get_intersection(dataAll_slice,
                                               min_samples=min_samples,
                                               eps=eps)

    return labels_2d


class Volume:

    def __init__(self, fullStack, segments):
        self.fullStack = fullStack
        self.segments = segments


    def get_volume(self, x=None, y=None, z=None):
        if x is None:
            x_min = 0
            x_max = self.segments.shape[0]-1
        else:
            x_min, x_max = x[0], x[1]

        if y is None:
            y_min = 0
            y_max = self.segments.shape[1]-1
        else:
            y_min, y_max = y[0], y[1]

        if z is None:
            z_min = 0
            z_max = self.segments.shape[2]-1
        else:
            z_min, z_max = z[0], z[1]

        vol = self.segments[x_min:x_max, y_min:y_max, z_min:z_max]

        return vol

    def get_segm_points(self, thresh, x=None, y=None, z=None):
        if x is None:
            x_min = 0
            x_max = self.segments.shape[0] - 1
        else:
            x_min, x_max = x[0], x[1]

        if y is None:
            y_min = 0
            y_max = self.segments.shape[1] - 1
        else:
            y_min, y_max = y[0], y[1]

        if z is None:
            z_min = 0
            z_max = self.segments.shape[2] - 1
        else:
            z_min, z_max = z[0], z[1]

        vol = self.segments[x_min:x_max, y_min:y_max, z_min:z_max]

        X, Y, Z = np.mgrid[:(x_max-x_min), :(y_max-y_min), :(z_max-z_min)]
        xyz = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
        mask = vol.flatten() > thresh

        return np.array(xyz[mask]), np.array(vol.flatten()[mask])


