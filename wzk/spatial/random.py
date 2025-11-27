import numpy as np
from wzk import spatial, geometry, random2, np2
from wzk.spatial.transform_2d import theta2dcm


def grid_quaternions(n3, flatten=False):
    s = np.linspace(0, 1, n3)
    theta1 = np.linspace(0, 2 * np.pi, n3, endpoint=False)
    theta2 = np.linspace(0, 2 * np.pi, n3, endpoint=False)

    s, theta1, theta2 = np.meshgrid(s, theta1, theta2)
    res = _sigma_theta2quaternions(s=s, theta1=theta1, theta2=theta2)
    if flatten:
        res = res.reshape(-1, 4)

    return res


def _sigma_theta2quaternions(s, theta1, theta2):
    sigma1 = np.sqrt(1 - s)
    sigma2 = np.sqrt(s)
    w = np.cos(theta2) * sigma2
    x = np.sin(theta1) * sigma1
    y = np.cos(theta1) * sigma1
    z = np.sin(theta2) * sigma2
    return np.stack([w, x, y, z], axis=-1)


def sample_quaternions(shape=None):
    """
    Effective Sampling and Distance Metrics for 3D Rigid Body Path Planning, James J. Kuffner (2004)
    https://ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf
    """
    s = np.random.random(shape)
    theta1 = np.random.uniform(0, 2 * np.pi, shape)
    theta2 = np.random.uniform(0, 2 * np.pi, shape)

    return _sigma_theta2quaternions(s=s, theta1=theta1, theta2=theta2)



def sample_dcm(shape=None):
    quat = sample_quaternions(shape=shape)
    return spatial.quaternions2dcm(quat=quat)


def sample_dcm_noise(shape=None, scale=0.01, mode="normal", n_dim=3):
    """
    samples rotation dcm where the absolute value of the rotation relates to 'scale' in rad
    """

    if n_dim == 3:
        rv = geometry.sample_points_on_sphere_3d(shape)
        rv *= random2.noise(shape=rv.shape[:-1], scale=scale, mode=mode)[..., np.newaxis]
        return spatial.rotvec2dcm(rotvec=rv)

    elif n_dim == 2:
        theta = np.random.uniform(low=0, high=2*np.pi, size=shape)
        if shape[-1] == 1:
            theta = theta[..., np.newaxis]

        return theta2dcm(theta=theta)

    else:
        raise ValueError(f"n_dim={n_dim} not supported, only [2, 3]")


def round_dcm(dcm, decimals=0):
    """Round dcm to degrees
    See numpy.round for more information
    decimals=+2: 123.456 -> 123.45
    decimals=+1: 123.456 -> 123.4
    decimals= 0: 123.456 -> 123.0
    decimals=-1: 123.456 -> 120.0
    decimals=-2: 123.456 -> 100.0
    """
    euler = spatial.dcm2euler(dcm)
    euler = np.rad2deg(euler)
    euler = np.round(euler, decimals=decimals)
    euler = np.deg2rad(euler)
    return spatial.euler2dcm(euler)


def sample_frames(x_low=np.zeros(3), x_high=np.ones(3), shape=None):
    if isinstance(x_low, (float, int)):
        x_low = np.ones(3) * x_low

    if isinstance(x_high, (float, int)):
        x_high = np.ones(3) * x_high

    assert len(x_low) == 3  # n_dim == 3
    assert len(x_high) == 3  # n_dim == 3

    return spatial.trans_quat2frame(trans=random2.random_uniform_ndim(low=x_low, high=x_high, shape=shape),
                                    quat=sample_quaternions(shape=shape))


def apply_noise(f, trans, rot, mode="normal"):
    n_dim = f.shape[-1] - 1
    s = tuple(np.array(np.shape(f))[:-2])

    f2 = f.copy()
    f2[..., :-1, -1] += random2.noise(shape=s + (n_dim,), scale=trans, mode=mode)
    f2[..., :-1, :-1] = f2[..., :-1, :-1] @ sample_dcm_noise(shape=s, scale=rot, mode=mode, n_dim=n_dim)
    return f2


def sample_around_f(f, trans, rot, mode="normal", shape=None):
    shape = np2.shape_wrapper(shape)
    f0 = np.zeros(shape+f.shape)
    f0[:] = f.copy()
    return apply_noise(f=f0, trans=trans, rot=rot, mode=mode)


def sample_frame_noise(trans, rot, shape=None, mode="normal"):
    f = spatial.initialize_frames(shape=shape, n_dim=3, mode="eye")
    return apply_noise(f=f, trans=trans, rot=rot, mode=mode)


def sample_frames_on_noisy_grid(x_grid, y_grid, z_grid,
                                f0, noise_trans, noise_rot,
                                n_samples):
    f_list = []
    n_total = len(x_grid) * len(y_grid) * len(z_grid)
    n_per_cell = int(np.ceil(n_samples / n_total))

    for x in x_grid:
        for y in y_grid:
            for z in z_grid:
                for n_noise in range(n_per_cell):
                    f = f0.copy()
                    f[..., :3, -1] = [x, y, z]
                    f = apply_noise(f, trans=noise_trans, rot=noise_rot)
                    f_list.append(f)

    f_list = np.array(f_list)
    f_list = f_list[np.random.choice(n_total*n_per_cell, n_samples, replace=False), :, :]
    return f_list
