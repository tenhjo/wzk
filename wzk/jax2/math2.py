from __future__ import annotations

import math
from itertools import product
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from wzk import ltd

from ._types import ArrayLike, AxisLike, float32, int32
from . import np2
from . import basics as b2
from . import shape as sh

GOLDEN_RATIO = jnp.asarray((jnp.sqrt(5.0) + 1) / 2, dtype=float32)


def make_monotonous_descending(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x).copy()
    for i in range(len(x)):
        x[i] = x[:i + 1].min()
    return x


def number2digits(num: int) -> list[int]:
    return [int(x) for x in str(num)]


def make_odd(x: ArrayLike, rounding: int = +1):
    return x - rounding * (x % 2 - 1)


def make_even(x: ArrayLike, rounding: int = +1):
    return x + rounding * (x % 2)


def make_even_odd(x: ArrayLike, mode: str, rounding: int = +1):
    mode = mode.lower()
    if mode in ("even", "e"):
        return make_even(x=x, rounding=rounding)
    if mode in ("odd", "o"):
        return make_odd(x=x, rounding=rounding)
    raise ValueError


def normalize_01(x: ArrayLike, low: ArrayLike | None = None, high: ArrayLike | None = None, axis: AxisLike = None):
    x = jnp.asarray(x)
    if low is None:
        low = jnp.min(x, axis=axis, keepdims=True)
    if high is None:
        high = jnp.max(x, axis=axis, keepdims=True)
    return (x - low) / (high - low)


def denormalize_01(x: ArrayLike, low: ArrayLike, high: ArrayLike):
    x = jnp.asarray(x)
    return x * (high - low) + low


def normalize11(x: ArrayLike, low: ArrayLike | None = None, high: ArrayLike | None = None, axis: AxisLike = None):
    x = jnp.asarray(x)
    if low is None:
        low = jnp.min(x, axis=axis, keepdims=True)
    if high is None:
        high = jnp.max(x, axis=axis, keepdims=True)
    return 2 * (x - low) / (high - low) - 1


def denormalize11(x: ArrayLike, low: ArrayLike, high: ArrayLike):
    x = jnp.asarray(x)
    return (x + 1) * (high - low) / 2 + low


def standardize_01(x: ArrayLike,
                   mean: ArrayLike | None,
                   std: ArrayLike | None,
                   axis: AxisLike = None):
    x = jnp.asarray(x)
    if mean is None:
        mean = jnp.mean(x, axis=axis, keepdims=True)
    if std is None:
        std = jnp.std(x, axis=axis, keepdims=True)
    return (x - mean) / std


def destandardize_01(x: ArrayLike, mean: ArrayLike, std: ArrayLike):
    x = jnp.asarray(x)
    return mean + x * std


def euclidean_norm(arr: ArrayLike, axis: int = -1, squared: bool = False):
    arr = jnp.asarray(arr)
    if squared:
        return (arr ** 2).sum(axis=axis)
    return jnp.sqrt((arr ** 2).sum(axis=axis))


def discretize(x: ArrayLike, step: float):
    if np.isinf(step) or np.isnan(step):
        return x

    x_arr = jnp.asarray(x)
    difference = jnp.mod(x_arr, step)

    if np.isscalar(x):
        diff = float(difference)
        x_scalar = float(np.asarray(x).item())
        if diff > step / 2:
            return x_scalar - (diff - step)
        return x_scalar - diff

    difference = jnp.where(difference > step / 2, difference - step, difference)
    return x_arr - difference


def dnorm_dx(x: ArrayLike, x_norm: ArrayLike | None = None):
    x = jnp.asarray(x)
    if x_norm is None:
        x_norm = jnp.linalg.norm(x, axis=-1)

    x_norm = jnp.asarray(x_norm)
    denom = x_norm[..., jnp.newaxis]
    return jnp.where(denom != 0, x / denom, x)


def dxnorm_dx(x: ArrayLike, return_norm: bool = False):
    x = jnp.asarray(x)
    n_dim = x.shape[-1]

    x_squared = x ** 2
    sq_sum = x_squared.sum(axis=-1, keepdims=True)
    inv = jnp.where(sq_sum > 0, sq_sum ** (-1.5), 0.0)

    outer = x[..., :, jnp.newaxis] * x[..., jnp.newaxis, :]
    jac = -outer
    diag = sq_sum - x_squared
    jac = jac.at[..., jnp.arange(n_dim), jnp.arange(n_dim)].set(diag)
    jac = jac * inv[..., jnp.newaxis]

    if return_norm:
        xn = jnp.where(sq_sum > 0, x / jnp.sqrt(sq_sum), x)
        return xn, jac
    return jac


def smooth_step(x: ArrayLike):
    x = jnp.asarray(x)
    res = -2 * x ** 3 + 3 * x ** 2
    return jnp.clip(res, 0, 1)


def smoother_step(x: ArrayLike):
    x = jnp.asarray(x)
    res = +6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3
    return jnp.clip(res, 0, 1)


def divisors(n: int, with_1_and_n: bool = False) -> list[int]:
    factors = {}
    nn = n
    i = 2
    while i * i <= nn:
        while nn % i == 0:
            factors[i] = factors.get(i, 0) + 1
            nn //= i
        i += 1
    if nn > 1:
        factors[nn] = 1

    primes = list(factors.keys())

    def generate(k):
        if k == len(primes):
            yield 1
        else:
            rest = generate(k + 1)
            prime = primes[k]
            for _factor in rest:
                prime_to_i = 1
                for _ in range(factors[prime] + 1):
                    yield _factor * prime_to_i
                    prime_to_i *= prime

    vals = list(generate(0))
    return vals if with_1_and_n else vals[1:-1]


def get_mean_divisor_pair(n: int) -> tuple[int, int]:
    assert isinstance(n, int)
    assert n >= 1

    div = divisors(n)
    if len(div) == 0:
        return 1, n

    div.sort()
    if len(div) % 2 == 1:
        idx_center = len(div) // 2
        return div[idx_center], div[idx_center]

    idx_center_plus1 = len(div) // 2
    idx_center_minus1 = idx_center_plus1 - 1
    return div[idx_center_minus1], div[idx_center_plus1]


def get_divisor(numerator: int | float, denominator: int | float) -> int:
    divisor = numerator / denominator
    divisor_int = int(divisor)
    assert divisor_int == divisor
    return divisor_int


def doubling_factor(small: float, big: float):
    return jnp.log2(big / small)


def modulo(x: ArrayLike, low: float, high: float):
    return (x - low) % (high - low) + low


def angle2minuspi_pluspi(x: ArrayLike):
    return modulo(x=x, low=-jnp.pi, high=+jnp.pi)


def log_b(x: ArrayLike, base: float = jnp.e):
    return jnp.log(x) / jnp.log(base)


def assimilate_orders_of_magnitude(a: ArrayLike, b: ArrayLike, base: int = 10):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    a_mean = jnp.abs(a).mean()
    b_mean = jnp.abs(b).mean()
    a_mean_log = jnp.log(a_mean)
    b_mean_log = jnp.log(b_mean)

    c = jnp.power(base, (a_mean_log + b_mean_log) / 2)

    aa = a * c / a_mean
    bb = b * c / b_mean

    return aa, bb, c


def rosenbrock2d(xy: ArrayLike, a: float = 1, b: float = 100):
    xy = jnp.asarray(xy)
    x, y = xy[..., 0], xy[..., 1]
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def d_rosenbrock2d(xy: ArrayLike, a: float = 1, b: float = 100):
    xy = jnp.asarray(xy)
    x, y = xy.T
    dx = -2 * (a - x) - 4 * b * (y - x ** 2) * x
    dy = 2 * b * (y - x ** 2)
    return jnp.concatenate([dx[..., jnp.newaxis], dy[..., jnp.newaxis]], axis=-1)


def bisection(f: Callable[[float], float],
              a: float,
              b: float,
              tol: float,
              max_depth: int = 50,
              verbose: int = 0,
              _depth: int = 0) -> float:
    assert a < b

    fa, fb = f(a), f(b)

    if np.sign(fa) == np.sign(fb):
        if (np.sign(fa) == +1 and fa < fb) or (np.sign(fa) == -1 and fa > fb):
            return bisection(f=f, a=a / 2, b=a, tol=tol, verbose=verbose, _depth=_depth + 1)
        return bisection(f=f, a=b, b=2 * b, tol=tol, verbose=verbose, _depth=_depth + 1)

    m = (a + b) / 2
    fm = f(m)

    if verbose > 0:
        print(f"depth {_depth}: a {a}, b {b}, m {m}, f(m) {fm}")

    if np.abs(fm) <= tol or _depth > max_depth:
        return m
    if np.sign(fa) == np.sign(fm):
        return bisection(f=f, a=m, b=b, tol=tol, verbose=verbose, _depth=_depth + 1)
    if np.sign(fb) == np.sign(fm):
        return bisection(f=f, a=a, b=m, tol=tol, verbose=verbose, _depth=_depth + 1)

    raise ValueError("Should not happen!")


def numeric_derivative(fun: Callable[..., ArrayLike],
                       x: ArrayLike,
                       eps: float = 1e-5,
                       axis: AxisLike = -1,
                       mode: str = "central",
                       diff: Callable[[ArrayLike, ArrayLike], ArrayLike] | None = None,
                       **kwargs_fun: Any):
    axis = sh.axis_wrapper(axis=axis, n_dim=np.ndim(x))

    x_np = np.array(x)
    f_x = np.asarray(fun(x, **kwargs_fun))
    fun_shape = np.shape(f_x)
    var_shape = ltd.atleast_tuple(np.array(np.shape(x_np))[(axis,)])
    derv = np.empty(fun_shape + var_shape)

    if diff is None:
        def diff(a, b):
            return a - b

    for idx in product(*(range(s) for s in var_shape)):
        eps_mat = np.zeros_like(x_np, dtype=float32)
        b2.insert(eps_mat, val=eps, idx=idx, axis=axis)

        if mode == "central":
            derv[(Ellipsis,) + idx] = diff(fun(x_np + eps_mat, **kwargs_fun),
                                           fun(x_np - eps_mat, **kwargs_fun)) / (2 * eps)

        elif mode == "forward":
            derv[(Ellipsis,) + idx] = diff(fun(x_np + eps_mat, **kwargs_fun), f_x) / eps

        elif mode == "backward":
            derv[(Ellipsis,) + idx] = diff(f_x, fun(x_np - eps_mat, **kwargs_fun)) / eps

        else:
            raise ValueError(f"Unknown mode {mode}")

    return jnp.asarray(derv)


def magic(n: int, m: int | None = None):
    if m is None:
        m = n

    shape = (n, m)
    n = int(max(n, m))

    if n < 1:
        raise ValueError("Size must be at least 1")
    if n == 1:
        mat = np.array([[1]])
    elif n == 2:
        mat = np.array([[1, 3],
                        [4, 2]])
    elif n % 2 == 1:
        p = np.arange(1, n + 1)
        mat = n * np.mod(p[:, None] + p - (n + 3) // 2, n) + np.mod(p[:, None] + 2 * p - 2, n) + 1
    elif n % 4 == 0:
        j = np.mod(np.arange(1, n + 1), 4) // 2
        k = j[:, None] == j
        mat = np.arange(1, n * n + 1, n)[:, None] + np.arange(n)
        mat[k] = n * n + 1 - mat[k]
    else:
        p = n // 2
        mat = magic(p)
        mat = np.block([[mat, mat + 2 * p * p], [mat + 3 * p * p, mat + p * p]])
        i = np.arange(p)
        k = (n - 2) // 4
        j = np.concatenate([np.arange(k), np.arange(n - k + 1, n)])
        mat[np.ix_(np.concatenate([i, i + p]), j)] = mat[np.ix_(np.concatenate([i + p, i]), j)]
        mat[np.ix_([k, k + p], [0, k])] = mat[np.ix_([k + p, k], [0, k])]

    mat = mat[:shape[0], :shape[1]]
    return jnp.asarray(mat, dtype=int32)


def binomial(n: int, k: int) -> int:
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def random_subset(n: int, k: int, m: int, dtype=int32):
    assert n == np.array(n, dtype=dtype)
    return jnp.asarray(np.array([np.random.choice(n, k, replace=False) for _ in range(m)]).astype(dtype), dtype=int32)


def irwin_hall_distribution(x: ArrayLike, n: int = 2):
    pre_factor = 1 / 2 / math.factorial(n - 1)

    f_xn = 0
    for k in range(n + 1):
        f_xn += (-1) ** k * binomial(n, k) * (x - k) ** (n - 1) * np.sign(x - k)

    return pre_factor * f_xn


__RCOND = 1e-5


def get_upper(n: int) -> np.ndarray:
    u = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if j > i:
                u[i, j] = True

    return u


def get_lower(n: int) -> np.ndarray:
    return get_upper(n=n).T


def project2null(A: ArrayLike,
                 x: ArrayLike,
                 clip: float | None = None,
                 clip_mode: str | None = None,
                 _rcond: float = __RCOND):
    x = np2.clip2(x, clip=clip, mode=clip_mode)

    A = jnp.asarray(A)
    x = jnp.asarray(x)
    AT = jnp.swapaxes(A, -2, -1)

    A0 = jnp.eye(A.shape[-1]) - (AT @ jnp.linalg.pinv(AT, rcond=_rcond))
    x0 = (A0 @ x[..., jnp.newaxis])[..., 0]

    x0 = np2.clip2(x0, clip=clip, mode=clip_mode)
    return x0


def solve_pinv(A: ArrayLike, b: ArrayLike, _rcond: float = __RCOND):
    try:
        A = jnp.asarray(A)
        b = jnp.asarray(b)
        x = (jnp.linalg.pinv(A, rcond=_rcond) @ b[..., jnp.newaxis])[..., 0]
    except Exception:  # noqa: BLE001
        x = np.zeros(np.shape(b)[:-1] + (np.shape(A)[-2],))
        return jnp.asarray(x)

    return x


def solve_lstsq(A: ArrayLike, b: ArrayLike, rcond: float | None = None):
    A = np.asarray(A)
    b = np.asarray(b)

    if A.ndim == 2 and b.ndim == 1:
        return jnp.asarray(np.linalg.lstsq(A, b, rcond=rcond)[0])

    if A.ndim == 3 and b.ndim == 2:
        nn, _, ny = A.shape
        x = np.zeros((nn, ny))
        for i in range(nn):
            x[i] = np.linalg.lstsq(A[i], b[i], rcond=rcond)[0]
        return jnp.asarray(x)

    raise ValueError


def solve_halley_damped(h: ArrayLike, j: ArrayLike, e: ArrayLike, damping: float):
    x = solve_cho_damped(A=j, b=e, damping=damping)
    hq = jnp.sum(jnp.asarray(h) * -x[..., jnp.newaxis, jnp.newaxis, :], axis=-1)
    j_hq = jnp.asarray(j) + 0.5 * hq
    x = solve_cho_damped(A=j_hq, b=e, damping=damping)
    return x


def solve_newton_damped(j: ArrayLike, e: ArrayLike, damping: float):
    return solve_cho_damped(A=j, b=e, damping=damping)


def solve_cho(A: ArrayLike, b: ArrayLike):
    A = np.asarray(A)
    b = np.asarray(b)

    if A.ndim == 2 and b.ndim == 1:
        return jnp.asarray(cho_solve(cho_factor(A), b))
    if A.ndim == 3 and b.ndim == 2:
        nn, _, ny = A.shape
        x = np.zeros((nn, ny))
        for i in range(nn):
            x[i] = cho_solve(cho_factor(A[i]), b[i])
        return jnp.asarray(x)

    raise ValueError("solve_cho: A and b must be 2D or 3D")


def solve_cho_damped(A: ArrayLike, b: ArrayLike, damping: float):
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    n, _ = A.shape[-2:]
    AT = jnp.swapaxes(A, -2, -1)
    AAT = A @ AT

    if damping > 0:
        AAT = AAT.at[..., range(n), range(n)].add(damping)

    x = AT @ solve_cho(np.asarray(AAT), np.asarray(b))[..., jnp.newaxis]
    x = x[..., 0]
    return x


def matrix_sqrt(A: ArrayLike):
    A = np.asarray(A)
    if A.ndim == 1:
        return jnp.sqrt(jnp.asarray(A))

    e_val, e_vec = np.linalg.eig(A)
    if (e_val < 0).any():
        raise ValueError("Matrix does not have a real square root")
    e_val_sqrt = np.sqrt(e_val)
    sqrt_A = e_vec @ np.diag(e_val_sqrt) @ np.linalg.inv(e_vec)
    return jnp.asarray(sqrt_A)


from wzk.math import math2 as _np_math2  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_np_math2, name)
