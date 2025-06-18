import numpy as np


def sample_24cube_rotations(n=None, i=None):
    if i is None:
        i = np.random.randint(low=0, high=24, size=n)

    return DCM24.copy()[i]


def xy2cube24(x: int, y: int):
    """
    face = [+-1, +-2, +-3]
    in which world axis does the face (x-axis) face

    clock = [++1, +-2, +3]
    in which world axis does the clock (y-axis) face

    (+x, +y) (+x, +z) (+x, -y) (+x, -z)
    (+y, +x) (+y, +z) (+y, -y) (+y, -z)
    (+z, +x) (+z, +y) (+z, -x) (+z, -y)
    (-x, -y) (-x, +z) (-x, -y) (-x, -y)
    (-y, +x) (-y, +z) (-y, -y) (-y, -z)
    (-z, +x) (-z, +y) (-z, -x) (-z, -y)

    short hand notation for the 24 cube orientations
    (direction of x axis, direction of y axis) -> third axis is then inferred by rhs
    """

    xyz = np.eye(3)
    dcm = np.eye(3)
    dcm[:, 0] = np.sign(x) * xyz[np.abs(x).astype(int) - 1]
    dcm[:, 1] = np.sign(y) * xyz[np.abs(y).astype(int) - 1]
    dcm[:, 2] = np.cross(dcm[:, 0], dcm[:, 1])
    return dcm


def get_all_24_cube_rotations():
    all_xy = [+1, +2, +3, -1, -2, -3]
    dcm24 = np.zeros((24, 3, 3))
    count = 0
    for x in all_xy:
        for y in all_xy:
            if y == x or y == -x:
                pass
            else:
                dcm24[count, :, :] = xy2cube24(x=x, y=y)
                count += 1
    return dcm24


DCM24 = get_all_24_cube_rotations()
