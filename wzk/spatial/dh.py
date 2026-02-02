import numpy as np

from wzk import math2


def dh2frame(q, d, theta, a, alpha):
    """Craig
    From wikipedia (https://en.wikipedia.org/wiki/Denavit–Hartenberg_parameters):
        d: offset along previous z to the common normal
        theta: angle about previous z, from old x to new x
        r: length of the common normal (aka "a", but if using this notation, do not confuse with alpha)
           Assuming a revolute joint, this is the radius about previous z
        alpha: angle about common normal, from old z axis to new z axis
    """

    cos_th = np.cos(theta + q)
    sin_th = np.sin(theta + q)
    cos_al = np.cos(alpha)
    sin_al = np.sin(alpha)

    return np.array([[cos_th, -sin_th, 0., a],
                     [cos_al * sin_th, cos_al * cos_th, -sin_al, -d * sin_al],
                     [sin_al * sin_th, sin_al * cos_th, cos_al, d * cos_al],
                     [0, 0, 0, 1]])


def dh2frame2(q, d, theta, a, alpha):

    cos_th = np.cos(theta + q)
    sin_th = np.sin(theta + q)
    cos_al = np.cos(alpha)
    sin_al = np.sin(alpha)

    frames = np.zeros((4, 4))

    frames[0, 0] = cos_th
    frames[0, 1] = -sin_th
    frames[0, 3] = a
    frames[1, 0] = cos_al * sin_th
    frames[1, 1] = cos_al * cos_th
    frames[1, 2] = -sin_al
    frames[1, 3] = -d * sin_al
    frames[2, 0] = sin_al * sin_th
    frames[2, 1] = sin_al * cos_th
    frames[2, 2] = cos_al
    frames[2, 3] = d * cos_al
    frames[3, 3] = 1
    return frames



def frame2dh(f):
    if f[0, 2] != 0:
        print("frame does not match DH formalism")
    theta = np.arctan2(-f[0, 1], f[0, 0])
    alpha = np.arctan2(-f[1, 2], f[2, 2])

    theta = math2.angle2minuspi_pluspi(theta)
    alpha = math2.angle2minuspi_pluspi(alpha)

    if np.cos(theta) != 0:
        d = f[2, 3] / np.cos(alpha)
    else:  # np.sin(theta) != 0:
        d = -f[1, 3] / np.sin(alpha)

    a = f[0, 3]
    return d, theta, a, alpha





def dh2frame_2d(q, theta, a):
    cos_th = np.cos(theta + q)
    sin_th = np.sin(theta + q)

    return np.array([[cos_th, -sin_th, a],
                     [sin_th, cos_th, 0],
                     [0, 0, 1]])
