import numpy as np
from scipy.spatial.transform import Rotation as R
from wzk import spatial, geometry


def sample_24cube_rotations(n=None, i=None):
    # 6 x 4
    rots = R.create_group("O")
    dcm24 = rots.as_matrix()
    # e24 = np.rad2deg(spatial.dcm2euler(dcm=dcm24, seq="zxz"))
    if i is None:
        i = np.random.randint(low=0, high=24, size=n)
    return dcm24[i]


def xy2cube24(x: int,  y: int):
    """
    face = [+-1, +-2, +-3]
    in which world axis does the face (x-axis) face

    clock = [++1, +-2, +3]
    in which world axis does the clock (y-axis) face

    (+x, +y) (+x, +z) (+x, -y) (+x, -z)
    (-x, -y) (-x, +z) (-x, -y) (-x, -y)
    (+y, +x) (+y, +z) (+y, -y) (+y, -z)
    (-y, +x) (-y, +z) (-y, -y) (-y, -z)
    (+z, +x) (+z, +y) (+z, -x) (+z, -y)
    (-z, +x) (-z, +y) (-z, -x) (-z, -y)
    """

    # TODO
    # short hand notation for the 24 cube orientations
    # (direction of x axis, direction of y axis) -> third axis is then infered by rhs

    xyz = np.eye(3)
    dcm = np.eye(3)
    dcm[:, 0] = np.sign(x) * xyz[np.abs(x).astype(int)-1]
    dcm[:, 1] = np.sign(y) * xyz[np.abs(y).astype(int)-1]
    dcm[:, 2] = np.cross(dcm[:, 0], dcm[:, 1])
    return dcm
