import numpy as np
from scipy.spatial.transform import Rotation

from wzk import ltd, np2, geometry, trajectory

from wzk.math.math2 import angle2minuspi_pluspi  # noqa
from wzk.spatial.util import initialize_frames, fill_frames_trans

# Nomenclature
# dcm ~ SE3 matrix (3x3)
# frame ~ (4x4) homogeneous matrix, SE3 + translation

__quat_scalar_first = True
__quat_canonical = True

# vectorized versions of scipy's Rotation.from_x().to_y()
def euler2dcm(euler: np.ndarray, seq="ZXZ"):
    """ZXZ == roll pitch yaw"""
    return Rotation.from_euler(seq, angles=euler.reshape((-1, 3)),
                               ).as_matrix().reshape(euler.shape[:-1] + (3, 3))  # noqa


def rpy2dcm(rpy: np.ndarray):
    return euler2dcm(euler=rpy, seq="ZXZ")


def quaternions2dcm(quat):
    return Rotation.from_quat(quat.reshape((-1, 4)), scalar_first=__quat_scalar_first,
                              ).as_matrix().reshape(quat.shape[:-1] + (3, 3))  # noqa


def rotvec2dcm(rotvec):
    return Rotation.from_rotvec(rotvec.reshape((-1, 3))
                                ).as_matrix().reshape(rotvec.shape[:-1] + (3, 3))  # noqa


def dcm2euler(dcm, seq="ZXZ"):
    return Rotation.from_matrix(dcm.reshape((-1, 3, 3))
                                ).as_euler(seq=seq).reshape(dcm.shape[:-2] + (3,))  # noqa


def dcm2quaternions(dcm):
    return Rotation.from_matrix(dcm.reshape((-1, 3, 3))
                                ).as_quat(canonical=__quat_canonical, scalar_first=__quat_scalar_first
                                          ).reshape(dcm.shape[:-2] + (4,))  # noqa


def dcm2rotvec(dcm):
    return Rotation.from_matrix(dcm.reshape((-1, 3, 3))
                                ).as_rotvec().reshape(dcm.shape[:-2] + (3,))  # noqa


# frame2rotation
# ----------------------------------------------------------------------------------------------------------------------
def frame2dcm(f):
    return f[..., :-1, :-1]


def frame2quat(f):
    return dcm2quaternions(f[..., :3, :3])


def frame2euler(f, seq="ZXZ"):
    return dcm2euler(f[..., :3, :3], seq=seq)


def frame2rotvec(f):
    return dcm2rotvec(f[..., :3, :3])


def frame2rotz(f):
    z_axis = f[..., :3, 2]
    a = np.arctan2(z_axis[..., 1], z_axis[..., 0])
    return a


# frame2trans_rot
# ----------------------------------------------------------------------------------------------------------------------
def frame2trans(f):
    return f[..., :-1, -1]


def frame2trans_dcm(f):
    return frame2trans(f), frame2dcm(f)


def frame2trans_rotvec(f):
    return frame2trans(f), frame2rotvec(f=f)


def frame2trans_quat(f):
    return frame2trans(f), frame2quat(f=f)


def frame2trans_euler(f, seq="ZXZ"):
    return frame2trans(f), frame2euler(f=f, seq=seq)


# 2frame
# ----------------------------------------------------------------------------------------------------------------------
def rotx2frame(alpha):
    if isinstance(alpha, (np.ndarray, tuple, list)):
        return np.array([rotx2frame(a) for a in alpha])

    return np.array([[1, 0, 0, 0],
                     [0, +np.cos(alpha), -np.sin(alpha), 0],
                     [0, +np.sin(alpha), +np.cos(alpha), 0],
                     [0, 0, 0, 1]])


def roty2frame(beta):
    if isinstance(beta, (np.ndarray, tuple, list)):
        return np.array([roty2frame(b) for b in beta])

    return np.array([[+np.cos(beta), 0, +np.sin(beta), 0],
                     [0, 1, 0, 0],
                     [-np.sin(beta), 0, +np.cos(beta), 0],
                     [0, 0, 0, 1]])


def rotz2frame(gamma):
    if isinstance(gamma, (np.ndarray, tuple, list)):
        return np.array([rotz2frame(g) for g in gamma])

    return np.array([[+np.cos(gamma), -np.sin(gamma), 0, 0],
                     [+np.sin(gamma), +np.cos(gamma), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def trans2frame(xyz=None, x=None, y=None, z=None):
    if xyz is not None:
        f = initialize_frames(shape=xyz.shape[:-1], n_dim=xyz.shape[-1], mode="eye")
        f[..., :-1, -1] = xyz
        return f

    n = np2.max_size(*ltd.remove_nones([x, y, z]))
    f = initialize_frames(shape=n, n_dim=3, mode="eye")
    if x is not None:
        f[..., 0, -1] = x
    if y is not None:
        f[..., 1, -1] = y
    if z is not None:
        f[..., 2, -1] = z

    return f


def trans_rot2frame(x, a, axis, squeeze=True):
    """translation and rotation around one axis -> frame"""
    x = np.atleast_2d(x)
    n = len(x)

    rv = np.zeros((n, 3))
    rv[:, axis] = a
    f = trans_rotvec2frame(trans=x, rotvec=rv)

    if squeeze and n == 1:
        f = np.squeeze(f)
    return f


def trans_quat2frame(trans=None, quat=None):
    s = np2.get_max_shape(trans, quat)

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    f[..., :-1, :-1] = quaternions2dcm(quat=quat)
    return f


def trans_rotvec2frame(trans=None, rotvec=None):
    s = np2.get_max_shape(trans, rotvec)

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    f[..., :-1, :-1] = rotvec2dcm(rotvec=rotvec)
    return f


def trans_euler2frame(trans: np.ndarray = None, euler: np.ndarray = None, seq: str = "ZXZ") -> np.ndarray:
    s = np2.get_max_shape(trans, euler)

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    f[..., :-1, :-1] = euler2dcm(euler=euler, seq=seq)
    return f


def vrotxyz2dcm(v_rotx, v_roty, v_rotz):
    s = np2.get_max_shape(v_rotx, v_roty, v_rotz)
    dcm = initialize_frames(shape=s[:-1], n_dim=2, mode="eye")

    if v_rotx is not None:
        dcm[..., :, 0] = v_rotx
    if v_roty is not None:
        dcm[..., :, 1] = v_roty
    if v_rotz is not None:
        dcm[..., :, 2] = v_rotz

    return dcm


def trans_vrotxyz2frame(trans, v_rotx, v_roty, v_rotz):
    dcm = vrotxyz2dcm(v_rotx=v_rotx, v_roty=v_roty, v_rotz=v_rotz)
    f = trans_dcm2frame(trans=trans, dcm=dcm)
    return f


def trans_dcm2frame(trans=None, dcm=None):
    s = np2.get_max_shape(trans, dcm)
    if dcm is not None:
        s = s[:-1]

    f = initialize_frames(shape=s[:-1], n_dim=3)
    fill_frames_trans(f=f, trans=trans)
    if dcm is not None:
        f[..., :-1, :-1] = dcm
    return f


# Sampling


# --- Sanity Checks ----------------------------------------------------------------------------------------------------
def is_rotation(r):
    assert np.shape(r)[-1] == 3
    _eps = 1e-5
    _squeeze = False
    if np.ndim(r) == 2:
        r = r[np.newaxis, :, :]
        _squeeze = True

    r[np.abs(r) < 1e-10] = 0
    b = np.array(np.sum(np.abs(r) > 2, axis=(-2, -1)) == 0, dtype=bool)
    b[b] = np.sum(np.abs(r[b]), axis=(-2, -1)) > _eps
    if np.any(b):
        r2 = r[b]
        rtr = r2 @ np.swapaxes(r2, -2, -1)
        b2 = np.linalg.det(rtr) > _eps
        b[b] = b2
        if np.any(b):
            b[b] = np.linalg.norm(dcm2rotvec(rtr[b2]), axis=(-1)) < _eps

    if _squeeze:
        b = b[0]

    return b


def is_frame(f):
    _eps = 1e-7
    a = is_rotation(f[..., :-1, :-1])
    b = np.abs(f[..., -1, -1] - 1) < _eps
    return np.logical_and(a, b)


# --- Linalg -----------------------------------------------------------------------------------------------------------
def get_frames_between(f0, f1, n):

    if np.ndim(f0) == 3 and np.ndim(f1) == 3:
        return np.array([get_frames_between(f0=f0_i, f1=f1_i, n=n) for (f0_i, f1_i) in zip(f0, f1)])

    x = trajectory.get_substeps(x=np.concatenate([f0[:-1, -1:], f1[:-1, -1:]], axis=1).T, n=n-1)

    dm = f0[:-1, :-1].T @ f1[:-1, :-1]
    rv = dcm2rotvec(dm)
    a = np.linalg.norm(rv)
    if np.allclose(a, 0):
        dm2 = np.zeros((n, 3, 3))
        dm2[:] = np.eye(3)
    else:
        rvn = rv/a
        a2 = np.linspace(0, a, n)
        rv2 = a2[:, np.newaxis] * rvn[np.newaxis, :]
        dm2 = rotvec2dcm(rv2)

    m = f0[:-1, :-1] @ dm2
    f = trans_dcm2frame(trans=x, dcm=m)
    return f


def get_mean_f(f):
    assert f.ndim == 3
    f_mean = f[0]
    for i in range(1, len(f)):
        f_mean = get_frames_between(f0=f_mean, f1=f[i], n=3)[1]

    return f_mean


def offset_frame(f, i=None, vm=None,
                 offset=0.01):
    """
    offset: in [m]
    i: idx along which axis to offset
    """
    assert (i is None) ^ (vm is None)

    f = f.copy()
    if i is not None:
        d = f[..., :-1, i]

    elif vm is None:
        d = f[..., :-1, :-1] @ vm

    else:
        raise RuntimeError("i and vm can not both be None")

    d = d * offset / np.linalg.norm(d, axis=-1, keepdims=True)
    f[..., :3, -1] -= d
    return f


def apply_f_or_none(f, f_or_none):
    if f_or_none is None or np.allclose(f_or_none, np.eye(f_or_none.shape[-1])):
        return f
    else:
        return f_or_none @ f


def invert(f):
    """
    Create the inverse of n homogenous trafo f.
    Assume n x n are the last two dimensions of the array
    """

    n_dim = f.shape[-1] - 1
    t = f[..., :n_dim, -1]  # Translation

    # Apply the inverse rotation on the translation
    f_inv = f.copy()
    f_inv[..., :n_dim, :n_dim] = np.swapaxes(f_inv[..., :n_dim, :n_dim], axis1=-1, axis2=-2)
    f_inv[..., :n_dim, -1:] = -f_inv[..., :n_dim, :n_dim] @ t[..., np.newaxis]
    return f_inv


def add_trans(f, x):
    f1 = f.copy()
    f1[..., :-1, -1] += x
    return f1


def make_x_hm(x):
    x1 = np.ones(x.shape[:-1] + (4,))
    x1[..., :3] = x[..., :3]
    return x1


def Ax(A, x):
    if A is None:
        return x

    if x.shape[-1] != A.shape[-1]:
        x = make_x_hm(x)

    if A.ndim == 2 and x.ndim == 2:
        A = A[np.newaxis, :, :]

    if A.ndim == 3 and x.ndim == 1:
        x = x[np.newaxis, :]

    b = np.sum(A * x[..., np.newaxis, :], axis=-1)[..., :-1]
    return b


def AxBxC(a, b, c):
    """
    a x (b x c) = (a.c) b  - (a.b) c
    x: cross product
    .: dot product
    """
    return (np.sum(a * c, axis=-1, keepdims=True) * b -
            np.sum(a * b, axis=-1, keepdims=True) * c)


def VxDCM(v, dcm):
    dcmT = np.cross(v[..., np.newaxis, :], np.swapaxes(dcm, -1, -2))
    return np.swapaxes(dcmT, -1, -2)


def try_cross_order():
    j0, j1, r = np.random.random((3, 3))

    r0 = np.cross(np.cross(j0, j1), r)
    r1 = np.cross(np.cross(j0, r), j1)
    r2 = np.cross(np.cross(j1, r), j0)
    tr = r0 - r1 + r2
    print(tr)


# --- Functions --------------------------------------------------------------------------------------------------------
def centroid_normal2f_plane(centroid, normal):
    dcm = np.eye(3, 3)
    dcm[2, :] = normal
    dcm[0, :] = geometry.get_orthonormal(normal)
    dcm2 = geometry.make_rhs(xyz=dcm, order=(2, 0))

    f_plane = trans_dcm2frame(trans=centroid, dcm=dcm2.T)
    return f_plane


def check_side_of_plane(o, u, v, x):
    n = np.cross(u, v)

    xyz = geometry.make_rhs(xyz=np.array([u, v, n]), order=(2, 0))

    f = trans_dcm2frame(trans=o, dcm=xyz.T)

    x = Ax(invert(f), x)

    side = x[:, 2] > 0
    return side


if __name__ == "__main__":
    pass


# FINDING angle axis representation is like an onion, the singularity is the boarder to the next 360 shell
#   0 is 360 degree away from the next singularity -> nice
