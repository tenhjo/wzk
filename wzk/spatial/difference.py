
from wzk.logger import log_print
import numpy as np

from wzk import printing
from wzk.spatial.transform import dcm2rotvec, frame2trans_dcm


def frame_logarithm(f0, f1, log_level=0):
    # https://github.com/CarletonABL/QuIK/blob/main/C%2B%2B/QuIK/IK/hgtDiff.cpp

    x0, dcm0 = frame2trans_dcm(f0)
    x1, dcm1 = frame2trans_dcm(f1)

    # x
    dx = x0 - x1

    # r
    ddcm = dcm0 @ np.swapaxes(dcm1, -2, -1)

    dr = np.zeros_like(dx)
    dr[..., 0] = ddcm[..., 2, 1] - ddcm[..., 1, 2]
    dr[..., 1] = ddcm[..., 0, 2] - ddcm[..., 2, 0]
    dr[..., 2] = ddcm[..., 1, 0] - ddcm[..., 0, 1]
    # [0, z2, y1]
    # [z1, 0, x2]
    # [y2, x1, 0]
    t = np.trace(ddcm, axis1=-2, axis2=-1)[..., np.newaxis]
    en = np.linalg.norm(dr, axis=-1, keepdims=True)

    b1 = np.logical_or(t > -.99, en > 1e-10)[..., 0]
    b2 = en[..., 0] < 1e-3
    b_large = ~b1
    b_small = np.logical_and(b1, b2)
    b_true = np.logical_and(b1, ~b2)

    dr[b_true] = np.arctan2(en[b_true], t[b_true] - 1) * dr[b_true] / en[b_true]
    dr[b_small] = (3 / 4 - t[b_small] / 12) * dr[b_small]
    dr[b_large] = np.pi * (ddcm[b_large].diagonal(axis1=-2, axis2=-1) + 1)

    if log_level > 0:
        printing.print_stats_bool(b=b_large, name="large")
        printing.print_stats_bool(b=b_small, name="small")
        printing.print_stats_bool(b=b_true, name="true")

    # combine
    log = np.concatenate([dx, dr], axis=-1)
    return log


# Location
# ----------------------------------------------------------------------------------------------------------------------
def location_difference(loc_a, loc_b):
    return np.linalg.norm(loc_a - loc_b, axis=-1)


def location_difference_cost(loc_a, loc_b):
    loc_diff = loc_a - loc_b
    lco_diff_sqrd = 0.5 * (loc_diff ** 2).sum(axis=-1)
    return loc_diff, lco_diff_sqrd


# Rotation
# ----------------------------------------------------------------------------------------------------------------------
def rotation_cost(rot):
    """
    Use cos(alpha) as a cost metric, it is symmetric around zero, and the derivative of arccos is not needed.

    # cos(theta) = (np.trace(rot, axis1=-2, axis2=-1) - n_dim + 2) / 2
    # cost = 1 - cos(theta)

    # theta = arccos(1-cost)
    # cost = 1 - (np.trace(rot, axis1=-2, axis2=-1) - n_dim + 2) / 2
    # cost = 1 - (np.trace(rot, axis1=-2, axis2=-1) - n_dim) / 2 + 1
    # cost = - (np.trace(rot, axis1=-2, axis2=-1) - n_dim) / 2
    # cost = (n_dim - np.trace(rot, axis1=-2, axis2=-1)) / 2
    """
    n_dim = rot.shape[-1]

    cost = (n_dim - np.trace(rot, axis1=-2, axis2=-1)) / 2
    return cost


def rotation_cost2dist(cost):
    return np.arccos(1 - cost)


def rotation_dist2cost(dist):
    return 1 - np.cos(dist)


def rotation_dist(rot):
    return rotation_cost2dist(cost=rotation_cost(rot=rot))


def rotation_difference(rot_a, rot_b):
    rot = rot_b @ rot_a.swapaxes(-2, -1)
    if rot.shape[-1] == 3:
        return np.linalg.norm(dcm2rotvec(rot), axis=-1)  # This is numerically more stable
    else:
        return np.arccos(rot[..., 0, 0])


def rotation_difference_cost(rot_a, rot_b):
    rot = rot_b @ rot_a.swapaxes(-2, -1)
    return rotation_cost(rot=rot)


# Combined
# ----------------------------------------------------------------------------------------------------------------------
def frame_difference(f_a, f_b, unit_trans="m", unit_rot="rad", log_level=0):
    loc = location_difference(loc_a=f_a[..., :-1, -1],
                              loc_b=f_b[..., :-1, -1])

    rot = rotation_difference(rot_a=f_a[..., :-1, :-1],
                              rot_b=f_b[..., :-1, :-1])

    if log_level > 0:
        log_print(f"Max : {np.max(loc*1000):.3f}mm, {np.max(np.rad2deg(rot)):.3f}deg")
        log_print(f"Mean: {np.mean(loc*1000):.3f}mm, {np.mean(np.rad2deg(rot)):.3f}deg")

    if unit_trans == "mm":
        loc *= 1000

    if unit_rot == "deg":
        rot = np.rad2deg(rot)

    return loc, rot


def frame_difference_cost(f_a, f_b):
    loc = location_difference_cost(loc_a=f_a[..., :-1, -1],
                                   loc_b=f_b[..., :-1, -1])[1]

    rot = rotation_difference_cost(rot_a=f_a[..., :-1, :-1],
                                   rot_b=f_b[..., :-1, :-1])
    return loc, rot
