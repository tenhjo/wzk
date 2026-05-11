import numpy as np
from numpy.typing import NDArray

MEAN_VEL = 0.5
MAX_VEL = 1.0
MAX_ACC = 1.5

__TIMESTEP = 0.001
__smoothstep_max_deriv = 1.875


def _smoothstep(x: NDArray) -> NDArray:
    x2 = np.clip(x, 0.0, 1.0)
    return 6.0 * x2**5 - 15.0 * x2**4 + 10.0 * x2**3


def _integral_smoothstep(x: NDArray) -> NDArray:
    x2 = np.clip(x, 0.0, 1.0)
    return x2**6 - 3.0 * x2**5 + 2.5 * x2**4


def _max_feasible_mean_vel(path_length: float, max_vel: float, max_acc: float) -> float:
    if path_length <= 0.0:
        return 0.0
    v_distance_limit = float(np.sqrt(path_length * max_acc / __smoothstep_max_deriv))
    v_cap = min(max_vel, v_distance_limit)
    return path_length / (path_length / v_cap + (__smoothstep_max_deriv * v_cap / max_acc))


def _solve_peak_vel(
    path_length: float,
    mean_vel: float,
    max_vel: float,
    max_acc: float,
) -> float:
    if path_length <= 0.0 or mean_vel <= 0.0:
        return 0.0

    mean_clamped = min(mean_vel, _max_feasible_mean_vel(path_length=path_length, max_vel=max_vel, max_acc=max_acc))

    # Solve:
    # mean = L / (L / v + k v / a), with k = max(d/dx smoothstep) = 1.875
    # -> k v^2 - (aL/mean) v + aL = 0
    quad_b = max_acc * path_length / mean_clamped
    disc = max(
        quad_b * quad_b - 4.0 * __smoothstep_max_deriv * max_acc * path_length,
        0.0,
    )
    v_peak = (quad_b - float(np.sqrt(disc))) / (2.0 * __smoothstep_max_deriv)

    v_distance_limit = float(np.sqrt(path_length * max_acc / __smoothstep_max_deriv))
    v_cap = min(max_vel, v_distance_limit)
    return float(np.clip(v_peak, 0.0, v_cap))


def _build_time_profile(
    path_length: float,
    v_peak: float,
    max_acc: float,
    timestep_size: float,
) -> NDArray:
    if path_length <= 0.0 or v_peak <= 0.0:
        return np.zeros((1,))

    ramp_time = __smoothstep_max_deriv * v_peak / max_acc
    ramp_distance = 0.5 * v_peak * ramp_time
    cruise_distance = max(path_length - 2.0 * ramp_distance, 0.0)
    cruise_time = cruise_distance / v_peak

    total_time = 2.0 * ramp_time + cruise_time
    n_steps = max(int(np.ceil(total_time / timestep_size)), 1)
    t = np.linspace(0.0, total_time, n_steps + 1)

    s = np.empty_like(t)
    up = t <= ramp_time
    cruise = (t > ramp_time) & (t <= ramp_time + cruise_time)
    down = t > (ramp_time + cruise_time)

    x_up = t[up] / ramp_time
    s[up] = v_peak * ramp_time * _integral_smoothstep(x=x_up)

    s[cruise] = ramp_distance + v_peak * (t[cruise] - ramp_time)

    tau = t[down] - (ramp_time + cruise_time)
    x_down = tau / ramp_time
    s[down] = ramp_distance + cruise_distance + v_peak * ramp_time * (x_down - _integral_smoothstep(x=x_down))
    s[-1] = path_length

    return s


def _resample_by_arc_length(q: NDArray, cum_arc_length: NDArray, s: NDArray) -> NDArray:
    q_out = np.empty((s.size, q.shape[1]))
    for i in range(q.shape[1]):
        q_out[:, i] = np.interp(x=s, xp=cum_arc_length, fp=q[:, i])
    return q_out


def slow_down_q_path(
    q: NDArray,
    mean_vel: float = MEAN_VEL,
    max_vel: float = MAX_VEL,
    max_acc: float = MAX_ACC,
    timestep: float = __TIMESTEP,
    weighting: NDArray | None = None,
) -> NDArray:
    assert 0.0 < mean_vel < max_vel
    assert 0.0 < max_vel
    assert 0.0 < max_acc

    if q.size == 0:
        return q
    if q.shape[0] < 2:
        return q

    dq = q[1:] - q[:-1]
    if weighting is not None:
        dq *= weighting
    dqn = np.linalg.norm(dq, axis=1)

    cum_length = np.concatenate([np.zeros((1,)), np.cumsum(dqn)])
    path_length = float(cum_length[-1])
    if path_length <= 0.0:
        return q[:1]

    v_peak = _solve_peak_vel(path_length=path_length, mean_vel=mean_vel, max_vel=max_vel, max_acc=max_acc)
    if v_peak <= 0.0:
        return q

    s = _build_time_profile(path_length=path_length, v_peak=v_peak, max_acc=max_acc, timestep_size=timestep)
    q_smoothed = _resample_by_arc_length(q=q, cum_arc_length=cum_length, s=s)
    return q_smoothed


def slow_down_between_two_q(
    q_start: np.ndarray,
    q_end: np.ndarray,
    max_vel: float,
    max_acc: float,
    timestep: float = __TIMESTEP,
    safety: float = 0.99,
) -> np.ndarray:
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    q = np.stack([q_start, q_end], axis=0)

    L = float(np.linalg.norm(q_start - q_end))
    if L <= 0.0:
        return q[:1]

    v_cap = min(max_vel, float(np.sqrt(L * max_acc / __smoothstep_max_deriv)))
    mean_vel = safety * L / (L / v_cap + __smoothstep_max_deriv * v_cap / max_acc)

    return slow_down_q_path(q=q, mean_vel=mean_vel, max_vel=max_vel, max_acc=max_acc, timestep=timestep)


def slow_down_between_two_q_list(
    q_list: np.ndarray,
    max_vel: float,
    max_acc: float,
    timestep: float = __TIMESTEP,
    safety: float = 0.99,
):

    q_path_list_smooth = []
    for _i, q_path in enumerate(q_list):
        path_smooth = slow_down_between_two_q(
            q_start=q_path[0], q_end=q_path[-1], max_vel=max_vel, max_acc=max_acc, timestep=timestep, safety=safety
        )
        q_path_list_smooth.append(path_smooth.tolist())
        print(len(path_smooth))
    return q_path_list_smooth


if __name__ == "__main__":
    # from wzk import mpl2

    q_start = np.array([0.0])
    q_end = np.array([1])
    q2 = slow_down_between_two_q(q_start=q_start, q_end=q_end, max_vel=0.6, max_acc=1.0, timestep=0.001)
    dq = q2[1:] - q2[:-1]
    ddq = dq[1:] - dq[:-1]

    print("samples:", q2.shape[0])
    print("peak vel:", float(np.max(np.abs(dq))) / __TIMESTEP)
    print("peak acc:", float(np.max(np.abs(ddq))) / (__TIMESTEP**2) if ddq.size else 0.0)

    # fig, ax = mpl2.new_fig(n_rows=3)
    # ax[0].plot(q2)
    # ax[1].plot(dq)
    # ax[2].plot(ddq)
    # fig.show()
