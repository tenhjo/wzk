import numpy as np

from wzk import math2, mpl2

from scipy.interpolate import interp1d


TIMESTEP_SEC = 0.001
RAMPS_SEC = 1.0
MEAN_VEL = 0.5  # rad/s
MAX_VEL = 1.0   # rad/s


# Ramps
def ramp_time2steps(delta, timestep_size):
    return int(np.ceil(delta / timestep_size))


def create_ramp(delta, timestep_size):
    n_ramp = ramp_time2steps(delta=delta, timestep_size=timestep_size)
    return math2.smoother_step(x=np.linspace(0, 1, n_ramp))


def add_ramps(a, ramp, verbose=0):

    n_ramp = len(ramp)
    n_ramp2 = min(n_ramp//2, len(a)//2)

    sum_ramp = sum(ramp)
    ramp_start = ramp * sum(a[:n_ramp2]) / sum_ramp
    ramp_end = ramp[::-1] * sum(a[-n_ramp2:]) / sum_ramp

    a_with_ramps = np.concatenate([ramp_start, a[n_ramp2:-n_ramp2], ramp_end])

    if verbose > 0:
        fig, axes = mpl2.new_fig(n_rows=2, title="a + s")
        axes[0].plot(np.arange(n_ramp2, n_ramp2+len(a)), a, c="r", label="Without Ramps")
        axes[0].plot(a_with_ramps, c="b", label="With Ramps")
        axes[0].legend()

        axes[1].plot(np.arange(n_ramp2, n_ramp2+len(a)), np.cumsum(a), c="r", label="Without Ramps")
        axes[1].plot(np.cumsum(a_with_ramps), c="b", label="With Ramps")

    return a_with_ramps


def calculate_slowing_factor(q, mean_vel, max_vel, timestep_size, weighting=None):

    q_steps = np.abs(q[1:, :] - q[:-1, :])

    weighting = 1 if weighting is None else weighting

    factor_mean = np.round(np.sqrt((q_steps**2 * weighting).sum(axis=-1))
                           / (timestep_size * mean_vel)).astype(int)
    factor_max = np.round(q_steps.max(axis=-1)
                          / (max_vel * timestep_size)).astype(int)

    factor = np.amax((factor_mean, factor_max), axis=0)
    factor = np.clip(factor, a_min=1, a_max=np.inf)

    return factor


def transform_slowing_factor(factor, verbose=0):
    """
    This slowing factor increases the waypoints not uniform along the whole path but
    adapts them depending on the factor indicating how much more points are needed for each segment of the path
    """
    factor = np.clip(factor, a_min=1, a_max=np.inf).astype(int)

    n0 = len(factor) + 1
    n1 = sum(factor)
    dt = 1/(n0-1)

    factor_cs = np.cumsum(np.hstack([[1], factor]))
    a = np.zeros(n1+1)
    a[factor_cs[:-1]] = dt

    for i in np.nonzero(factor != 1)[0]:
        a[factor_cs[i]:factor_cs[i+1]] = dt / factor[i]

    assert np.isclose(sum(a), 1)

    if verbose > 0:
        fig, axes = mpl2.new_fig(n_rows=2)
        axes[0].plot(a, c="b")
        axes[1].plot(np.cumsum(a), c="b")
    return a


def warp_time(x, s, is_vel, order=1):
    """
    Always perform linear interpolation otherwise it the trajectories might no longer be consistent
    """

    s = np.clip(s, 0, 1)

    n_wp, n_dof = x.shape
    n_wp_new = len(s)
    xp = np.linspace(start=0, stop=1, num=n_wp)
    q2 = np.empty((n_wp_new, n_dof))

    # Split this cases because numpy is 10x faster than scipy for the linear case
    if order != 1:
        def interpolate(fp):
            return np.interp(x=s, xp=xp, fp=fp)

    else:
        def interpolate(fp):
            return interp1d(x=xp, y=fp, kind=order)(s)

    for i, q_i in enumerate(x.T):

        if is_vel is not None and is_vel[i]:
            # You can not interpolate velocities
            # -> integrate to positions, interpolate, differentiate
            q_i = np.cumsum(q_i)
            q_i = interpolate(q_i)
            q_i = np.diff(np.concatenate([q_i[:1], q_i]))

        else:
            q_i = interpolate(q_i)

        q2[:, i] = q_i

    return q2


def warp_time_with_ramps(q, a, ramp, is_vel):
    a = add_ramps(a, ramp=ramp)
    s = np.cumsum(a)
    return warp_time(x=q, s=s, is_vel=is_vel)


def slow_down_q_path(q,
                     mean_vel=MEAN_VEL, max_vel=MAX_VEL,
                     ramps_sec=RAMPS_SEC, timestep_size=TIMESTEP_SEC,
                     weighting=None,):
    if mean_vel == -1:
        return q

    if np.size(q) == 0:
        return q

    if np.allclose(q[0], q[-1], atol=1e-5, rtol=1e-5):
        return q[:1, :]

    factor = calculate_slowing_factor(q=q, mean_vel=mean_vel, max_vel=max_vel,
                                      timestep_size=timestep_size,
                                      weighting=weighting)

    a = transform_slowing_factor(factor=factor)

    ramp = create_ramp(delta=ramps_sec, timestep_size=timestep_size)
    return warp_time_with_ramps(q=q, a=a, ramp=ramp, is_vel=None)


def slow_down_q_path_list(q_path_list,
                          timestep_sec: float = TIMESTEP_SEC,
                          ramps_sec: float = RAMPS_SEC,
                          mean_vel: float = 0.5,           # rad/s
                          max_vel: float = 1,              # rad/s
                          verbose=1):

    q_path_list_smooth = []
    for i, q_path in enumerate(q_path_list):
        path_smooth = slow_down_q_path(q=q_path, mean_vel=mean_vel, max_vel=max_vel,
                                       ramps_sec=ramps_sec, timestep_size=timestep_sec)
        if len(path_smooth) < int((ramps_sec / timestep_sec) * 3):
            path_smooth = slow_down_q_path(q=q_path, mean_vel=mean_vel*1.5, max_vel=max_vel,
                                           ramps_sec=ramps_sec/2, timestep_size=timestep_sec)

        q_path_list_smooth.append(path_smooth.tolist())
        if verbose > 0:
            print(f"{i} | shape: {path_smooth.shape}")

    return q_path_list_smooth


def test_calculate_slowing_factor():
    q = np.random.random((10, 3))
    mean_vel = 0.1
    max_vel = 1
    calculate_slowing_factor(q=q, mean_vel=mean_vel, max_vel=max_vel, timestep_size=0.001)


if __name__ == '__main__':
    test_calculate_slowing_factor()
