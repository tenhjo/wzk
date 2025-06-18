"""
Original scripy by Clay Fannigan. Improvement by Max Bazik, with scaling added
by Alvin Wan, per:

Scaling iterative closest point algorithm for registration of m–D point sets
 - Du et al. (https://doi.org/10.1016/j.jvcir.2010.02.005)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B, scaling=False):
    """
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector for translation
      s: 3x1 column vector for scaling
    """

    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # compute scaling
    if scaling:
        s = sum(b.T.dot(a) for a, b in zip(AA, BB)) / sum(a.T.dot(a) for a in AA)
    else:
        s = 1.0

    # translation
    t = centroid_B.T - s * np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[:-1, :-1] = R
    T[:-1, -1] = t

    return T, R, t, s


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=100, tolerance=1e-10):
    """
    The Iterative Closest Point method
    Input:
        A: Nx3 numpy array of source 3D points
        B: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    """

    # make points homogeneous, copy them to maintain the originals
    n, n_dim = A.shape
    src = np.ones((n_dim+1, A.shape[0]))
    dst = np.ones((n_dim+1, B.shape[0]))
    src[:-1, :] = np.copy(A.T)
    dst[:-1, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    try:
        distances = np.inf

        for i in range(max_iterations):
            # find the nearest neighbours between the current source and destination points
            distances, indices = nearest_neighbor(src[:-1, :].T, dst[:-1, :].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _, s = best_fit_transform(src[:-1, :].T, dst[:-1, indices].T)

            # update the current source
            src = T.dot(src) * s

            # check error
            mean_error = np.sum(distances) / distances.size
            if abs(prev_error-mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T, _, _, s = best_fit_transform(A, src[:-1, :].T)

        return T, s, distances

    except (ValueError, np.linalg.LinAlgError) as e:
        print(e)
        return np.eye(n_dim+1), 1, np.array([np.inf])


def try_random():
    from wzk import mpl2, spatial
    n = 10
    x0 = np.random.random((n, 3))
    x1 = x0 + 0.1

    A, s, d = icp(A=x0.copy(), B=x1.copy())

    print(A)
    A = spatial.invert(A)
    x11 = spatial.Ax(A=A, x=x1)
    x0 = x0[:, :2]
    x11 = x11[:, :2]

    fig, ax = mpl2.new_fig()
    ax.plot(*x0.T, marker="o", color="red", ls="")
    ax.plot(*x11.T, marker="x", color="blue", ls="")


if __name__ == "__main__":
    try_random()
