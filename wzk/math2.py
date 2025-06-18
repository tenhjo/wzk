import math
from itertools import product

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from wzk import np2, ltd

# a/b = (a+b) / a -> a / b =
GOLDEN_RATIO = (np.sqrt(5.0) + 1) / 2


def make_monotonous_descending(x):
    for i in range(len(x)):
        x[i] = x[:i+1].min()
    return x


def number2digits(num):
    return [int(x) for x in str(num)]


def make_odd(x, rounding=+1):
    return x - rounding * (x % 2 - 1)


def make_even(x, rounding=+1):
    return x + rounding * (x % 2)


def make_even_odd(x, mode, rounding=+1):
    if mode.lower() == "even" or mode.lower() == "e":
        return make_even(x=x, rounding=rounding)
    elif mode.lower() == "odd" or mode.lower() == "o":
        return make_odd(x=x, rounding=rounding)
    else:
        raise ValueError


# Normalize
def normalize_01(x, low=None, high=None, axis=None):
    """
    Normalize [low, high] to [0, 1]
    low and high should either be scalars or have the same dimension as the last dimension of x
    """

    if low is None:
        low = np.min(x, axis=axis, keepdims=True)

    if high is None:
        high = np.max(x, axis=axis, keepdims=True)

    return (x-low) / (high-low)


def denormalize_01(x, low, high):
    return x * (high - low) + low


def normalize11(x, low=None, high=None, axis=None):
    """
    Normalize [low, high] to [-1, 1]
    low and high should either be scalars or have the same dimension as the last dimension of x
    """
    if low is None:
        low = np.min(x, axis=axis, keepdims=True)

    if high is None:
        high = np.max(x, axis=axis, keepdims=True)

    return 2 * (x - low) / (high - low) - 1


def denormalize11(x, low, high):
    """
    Denormalize [-1, 1] to [low, high]
    low and high should either be scalars or have the same dimension as the last dimension of x
    """
    return (x + 1) * (high - low)/2 + low


# Standardize
def standardize_01(x, mean, std, axis=None):
    if mean is None:
        mean = np.mean(x, axis=axis, keepdims=True)

    if std is None:
        std = np.std(x, axis=axis, keepdims=True)

    return (x-mean) / std


def destandardize_01(x, mean, std):
    return mean + x*std


def euclidean_norm(arr, axis=-1, squared=False):
    if squared:
        return (arr**2).sum(axis=axis)
    else:
        return np.sqrt((arr**2).sum(axis=axis))


def discretize(x, step):

    if np.isinf(step) or np.isnan(step):
        return x

    difference = x % step  # distance to the next discrete value

    if isinstance(x, (int, float)):
        if difference > step / 2:
            return x - (difference - step)
        else:
            return x - difference

    else:
        difference[difference > step / 2] -= step  # round correctly
        return x - difference


def dnorm_dx(x, x_norm=None):
    """ ∂ |x| / ∂ x
     normalization over last dimension
     """
    if x_norm is None:
        x_norm = np.linalg.norm(x, axis=-1)

    dn_dx = x.copy()
    i = x_norm != 0  # All steps where there is movement between t, t+1
    dn_dx[i, :] = dn_dx[i, :] / x_norm[i][..., np.newaxis]
    return dn_dx


def dxnorm_dx(x, return_norm=False):
    """
    ∂ (x/|x|) / ∂ x
    normalization over last dimension

    Calculate Jacobian
      xn       =           x * (x^2 + y^2 + z^2)^(-1/2)
    d xn / d x = (y^2 + z^2) * (x^2 + y^2 + z^2)^(-3/2)
    d yn / d y = (x^2 + y^2) * (x^2 + y^2 + z^2)^(-3/2)
    d zn / d z=  (x^2 + z^2) * (x^2 + y^2 + z^2)^(-3/2)

    Pattern of numerator
    X123
    0X23
    01X3
    012X

    d xn / d y = -(x*y) * (x^2 + y^2 + z^2)^(-3/2)
    d xn / d z = -(x*z) * (x^2 + y^2 + z^2)^(-3/2)

    jac = [[dxn/dx, dxn/dy, dxn/dz]
           [dyn/dx, dyn/dy, dyn/dz]
           [dzn/dx, dzn/dy, dzn/dz]

    """

    n_dim = x.shape[-1]

    off_diag_idx = [[j for j in range(n_dim) if i != j] for i in range(n_dim)]

    dxn_x = np.empty(x.shape + x.shape[-1:])
    x_squared = x**2

    # Diagonal
    dxn_x[..., np.arange(n_dim), np.arange(n_dim)] = x_squared[..., off_diag_idx].sum(axis=-1)

    # Off-Diagonal
    dxn_x[..., np.arange(n_dim)[..., np.newaxis], off_diag_idx] = -x[..., np.newaxis] * x[..., off_diag_idx]

    dxn_x *= (x_squared.sum(axis=-1, keepdims=True)**(-3/2))[..., np.newaxis]

    if return_norm:
        x /= np.sqrt(x_squared.sum(axis=-1, keepdims=True))
        return x, dxn_x
    else:
        return dxn_x


# Smooth
def smooth_step(x):
    """
    https://en.wikipedia.org/wiki/Smoothstep
    Interpolation which has zero 1st-order derivatives at x = 0 and x = 1,
     ~ cubic Hermite interpolation with clamping.
    """
    res = -2 * x**3 + 3 * x**2
    return np.clip(res, 0, 1)


def smoother_step(x):
    """
    https://en.wikipedia.org/wiki/Smoothstep+
    Ken Perlin suggests an improved version of the smooth step function,
    which has zero 1st- and 2nd-order derivatives at x = 0 and x = 1
    """
    res = +6 * x**5 - 15 * x**4 + 10 * x**3
    return np.clip(res, 0, 1)


# Divisors
def divisors(n, with_1_and_n=False):
    """
    https://stackoverflow.com/questions/171765/what-is-the-best-way-to-get-all-the-divisors-of-a-number#171784
    """

    # Get factors and their counts
    factors = {}
    nn = n
    i = 2
    while i*i <= nn:
        while nn % i == 0:
            if i not in factors:
                factors[i] = 0
            factors[i] += 1
            nn //= i
        i += 1
    if nn > 1:
        factors[nn] = 1

    primes = list(factors.keys())

    # Generate factors from primes[k:] subset
    def generate(k):
        if k == len(primes):
            yield 1
        else:
            rest = generate(k+1)
            prime = primes[k]
            for _factor in rest:
                prime_to_i = 1
                # Prime_to_i iterates prime**o values, o being all possible exponents
                for _ in range(factors[prime] + 1):
                    yield _factor * prime_to_i
                    prime_to_i *= prime

    if with_1_and_n:
        return list(generate(0))
    else:
        return list(generate(0))[1:-1]


def get_mean_divisor_pair(n):
    """
    Calculate the 'mean' pair of divisors. The two divisors should be as close as possible to the sqrt(n).
    The smaller divisor is the first value of the output pair
    10 -> 2, 5
    20 -> 4, 5
    24 -> 4, 6
    25 -> 5, 5
    30 -> 5, 6
    40 -> 5, 8
    """
    assert isinstance(n, int)
    assert n >= 1

    div = divisors(n)
    if len(div) == 0:
        return 1, n

    div.sort()

    # if numbers of divisors is odd -> n = o * o : power number
    if len(div) % 2 == 1:
        idx_center = len(div) // 2
        return div[idx_center], div[idx_center]

    # else get the two numbers at the center
    else:
        idx_center_plus1 = len(div) // 2
        idx_center_minus1 = idx_center_plus1 - 1
        return div[idx_center_minus1], div[idx_center_plus1]


def get_divisor(numerator, denominator):
    divisor = numerator / denominator
    divisor_int = int(divisor)

    assert divisor_int == divisor
    return divisor_int


def doubling_factor(small, big):
    return np.log2(big / small)


def modulo(x, low, high):
    return (x - low) % (high - low) + low


def angle2minuspi_pluspi(x):
    return modulo(x=x, low=-np.pi, high=+np.pi)
    # modulo is faster for larger arrays, for small ones they are similar but arctan is faster in this region
    #  -> as always you have to make a trade-off
    # return np.arctan2(np.sin(x), np.cos(x))


def log_b(x, base=np.e):
    # https://stackoverflow.com/questions/25169297/numpy-logarithm-with-base-n
    return np.log(x) / np.log(base)


def assimilate_orders_of_magnitude(a, b, base=10):
    a_mean = np.abs(a).mean()
    b_mean = np.abs(b).mean()
    np.log1p()
    a_mean_log = np.log(a_mean)
    b_mean_log = np.log(b_mean)

    c = np.power(base, (a_mean_log + b_mean_log) / 2)

    aa = a * c / a_mean
    bb = b * c / b_mean

    return aa, bb, c


# Functions
def rosenbrock2d(xy, a=1, b=100):
    # https://en.wikipedia.org/wiki/Rosenbrock_function
    # Minimum f(a, a**2) = 0
    xy = np.array(xy)
    x, y = xy[..., 0], xy[..., 1]
    return (a - x)**2 + b*(y - x**2)**2


def d_rosenbrock2d(xy, a=1, b=100):
    xy = np.array(xy)
    x, y = xy.T
    dx = -2*(a-x) - 4*b*(y-x**2)*x
    dy =          + 2*b*(y-x**2)  # noqa
    return np.concatenate([dx[..., np.newaxis], dy[..., np.newaxis]], axis=-1)


def bisection(f, a, b, tol, max_depth=50, verbose=0, __depth=0,):
    """
    aka binary search

    https://pythonnumericalmethods.berkeley.edu/notebooks/chapter19.03-Bisection-Method.html
    Approximates a root of f bounded by a and b to within tolerance
    | f(m) | < tol with m the midpoint between a and b.

    Recursive implementation
    """
    # check if a and b bound a root
    assert a < b

    fa, fb = f(a), f(b)

    # HEURISTIC
    if np.sign(fa) == np.sign(fb):
        print(f"The scalars a {a} and b {b} do not bound a root.\n"
              f"A heuristic is tried to shift the limits, but this is not guaranteed to work; "
              f"only if the function is monotonic.\n"
              f"Check the limits again manually!.")

        if (np.sign(fa) == +1 and fa < fb) or (np.sign(fa) == -1 and fa > fb):
            return bisection(f=f, a=a/2, b=a, tol=tol, verbose=verbose, __depth=__depth + 1)
        else:
            return bisection(f=f, a=b, b=2*b, tol=tol, verbose=verbose, __depth=__depth + 1)

        # else:
        #     if fa < fb:
        #         bisection(f=f, a=b, b=2*b, tol=tol)
        #     else:
        #         bisection(f=f, a=a/2, b=a, tol=tol)

    # get midpoint
    m = (a + b) / 2
    fm = f(m)

    if verbose > 0:
        print(f"depth {__depth}: a {a}, b {b}, m {m}, f(m) {fm}")

    if np.abs(fm) <= tol or __depth > max_depth:  # stopping condition, report m as root
        return m

    elif np.sign(fa) == np.sign(fm):  # m is an improvement on a
        return bisection(f=f, a=m, b=b, tol=tol, verbose=verbose, __depth=__depth+1)

    elif np.sign(fb) == np.sign(fm):  # m is an improvement on b
        return bisection(f=f, a=a, b=m, tol=tol, verbose=verbose, __depth=__depth + 1)


# Derivative
def numeric_derivative(fun, x, eps=1e-5, axis=-1, mode="central",
                       diff=None,
                       **kwargs_fun):
    """
    Use central, forward or backward difference scheme to calculate the
    numeric derivative of function at point x.
    'axis' indicates the dimensions of the free variables.
    The result has the shape f(x).shape + x.shape[axis]
    """
    axis = np2.axis_wrapper(axis=axis, n_dim=np.ndim(x))

    f_x = fun(x, **kwargs_fun)
    fun_shape = np.shape(f_x)
    var_shape = ltd.atleast_tuple(np.array(np.shape(x))[(axis,)])
    eps_mat = np.empty_like(x, dtype=float)

    derv = np.empty(fun_shape + var_shape)

    if diff is None:
        def diff(a, b):
            return a - b

    def update_eps_mat(_idx):
        eps_mat[:] = 0
        np2.insert(eps_mat, val=eps, idx=_idx, axis=axis)

    for idx in product(*(range(s) for s in var_shape)):
        update_eps_mat(_idx=idx)

        if mode == "central":
            derv[(Ellipsis,) + idx] = diff(fun(x + eps_mat, **kwargs_fun),
                                           fun(x - eps_mat, **kwargs_fun)) / (2 * eps)

        elif mode == "forward":
            derv[(Ellipsis, ) + idx] = diff(fun(x + eps_mat, **kwargs_fun), f_x) / eps

        elif mode == "backward":
            derv[(Ellipsis, ) + idx] = diff(f_x, fun(x - eps_mat, **kwargs_fun)) / eps

    return derv


# Magic
def magic(n, m=None):
    """
    Equivalent of the MATLAB function:
    M = magic(n) returns an n-by-n matrix constructed from the integers 1 through n2 with equal row and column sums.
    https://stackoverflow.com/questions/47834140/numpy-equivalent-of-matlabs-magic

    when a rectangle shape is given, the function returns just the sub-matrix. here the original properties do no
    longer hold.
    """

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
        p = np.arange(1, n+1)
        mat = n*np.mod(p[:, None] + p - (n+3)//2, n) + np.mod(p[:, None] + 2*p-2, n) + 1
    elif n % 4 == 0:
        j = np.mod(np.arange(1, n+1), 4) // 2
        k = j[:, None] == j
        mat = np.arange(1, n*n+1, n)[:, None] + np.arange(n)
        mat[k] = n*n + 1 - mat[k]
    else:
        p = n//2
        mat = magic(p)
        mat = np.block([[mat, mat+2*p*p], [mat+3*p*p, mat+p*p]])
        i = np.arange(p)
        k = (n-2)//4
        j = np.concatenate([np.arange(k), np.arange(n-k+1, n)])
        mat[np.ix_(np.concatenate([i, i+p]), j)] = mat[np.ix_(np.concatenate([i+p, i]), j)]
        mat[np.ix_([k, k+p], [0, k])] = mat[np.ix_([k+p, k], [0, k])]

    mat = mat[:shape[0], :shape[1]]
    return mat


# Clustering
def k_farthest_neighbors(x, k, weighting=None, mode="inverse_sum"):
    n = len(x)
    eps = 1e-6
    m_dist = x[np.newaxis, :, :] - x[:, np.newaxis, :]
    weighting = np.ones(x.shape[-1]) if weighting is None else weighting
    m_dist = ((m_dist * weighting)**2).sum(axis=-1)

    cum_dist = m_dist.sum(axis=-1)

    idx = np.array([np.argmax(cum_dist)])

    for i in range(k-1):
        m_dist_cur = m_dist[idx]

        if mode == "inverse_sum":
            m_dist_cur[np.arange(i+1), idx] = 1
            obj = -np.sum(1/(m_dist_cur+eps), axis=0)

        elif mode == "sum":
            obj = np.sum(m_dist_cur, axis=0)

        else:
            raise ValueError(f"Unknown mode {mode}")

        # C)
        # m_dist_cur_sum = m_dist_cur.sum(axis=0)
        # m_dist_cur_std = np.std(m_dist_cur, axis=0)
        # obj = m_dist_cur_sum + (m_dist_cur_std.max(initial=0) - m_dist_cur_std) * 1000

        # D
        # m_dist_cur[np.arange(i+1), idx] = np.inf
        # jj = np.argmin(m_dist_cur, axis=1)
        # kk = np.argmax(m_dist_cur[np.arange(i+1), jj])
        # idx = np.hstack([idx, [jj[kk]]]).astype(int)
        # continue

        idx_new = np.argsort(obj)[::-1]
        for j in range(n):
            if idx_new[j] not in idx:
                idx = np.hstack([idx, [idx_new[j]]]).astype(int)
                break

    return idx


def vis_k_farthest_neighbors():
    x = np.random.random((500, 2))
    k = 5
    idx = k_farthest_neighbors(x=x, k=k)

    from wzk import new_fig
    fig, ax = new_fig(aspect=1)
    ax.plot(*x.T, ls="", marker="o", color="b", markersize=5, alpha=0.5)
    ax.plot(*x[idx, :].T, ls="", marker="x", color="r", markersize=10)


# Combinatorics
def binomial(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def random_subset(n, k, m, dtype=np.uint16):
    assert n == np.array(n, dtype=dtype)
    return np.array([np.random.choice(n, k, replace=False) for _ in range(m)]).astype(np.uint16)


def irwin_hall_distribution(x, n=2):
    """
    https://en.wikipedia.org/wiki/Irwin-Hall_distribution
    """

    pre_factor = 1 / 2 / math.factorial(n - 1)

    f_xn = 0
    for k in range(n + 1):
        f_xn += (-1) ** k * binomial(n, k) * (x - k) ** (n - 1) * np.sign(x - k)

    return pre_factor * f_xn


def test_dxnorm_dx():
    x = np.vstack([magic(3).T]*3)
    j = dxnorm_dx(x)
    print(j[0])


# ----------------------------------------------------------------------------------------------------------------------
# Linear Algebra
__RCOND = 1e-5


def get_upper(n):
    u = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if j > i:
                u[i, j] = True

    return u


def get_lower(n):
    return get_upper(n=n).T


def project2null(A, x, clip=None, clip_mode=None, __rcond=__RCOND):
    """
    Clipping happens before and after the projection step.
    If the determinant of the projection is not larger than 1, the second clipping has no effect.
    """
    x = np2.clip2(x, clip=clip, mode=clip_mode)
    
    AT = np.swapaxes(A, -2, -1)

    A0 = np.eye(A.shape[-1]) - (AT @  np.linalg.pinv(AT, rcond=__rcond))
    x0 = (A0 @ x[..., np.newaxis])[..., 0]

    # same as:
    # n, m = A.shape[-2:]
    # A_big = np.block([[np.eye(m), AT], [A, np.zeros((n, n))]])
    # b_big = np.zeros(x.shape[:-1] + (n+m,))
    # b_big[..., :m] = x
    # x0_cho = solve_cho(A=A_big, b=b_big)

    x0 = np2.clip2(x0, clip=clip, mode=clip_mode)  # this is only for safety, normally without effect
    return x0


def solve_pinv(A, b, __rcond=__RCOND):
    try:
        x = (np.linalg.pinv(A, rcond=__rcond) @ b[..., np.newaxis])[..., 0]

    except np.linalg.LinAlgError:
        print("solve_pinv: np.linalg.LinAlgError")
        x0 = np.zeros(b.shape[:-1] + (A.shape[-2],))
        return x0
    
    return x


def solve_lstsq(A, b, rcond=None):

    if A.ndim == 2 and b.ndim == 1:
        return np.linalg.lstsq(A, b, rcond=rcond)[0]

    elif A.ndim == 3 and b.ndim == 2:
        nn, nx, ny = A.shape
        x = np.zeros((nn, ny))
        for i in range(nn):
            x[i] = np.linalg.lstsq(A[i], b[i], rcond=rcond)[0]
        return x
    else:
        raise ValueError


def solve_halley_damped(h, j, e, damping):
    x = solve_cho_damped(A=j, b=e, damping=damping)
    hq = np.sum(h * -x[..., np.newaxis, np.newaxis, :], axis=-1)
    j_hq = j + 0.5 * hq
    x = solve_cho_damped(A=j_hq, b=e, damping=damping)
    return x


def solve_newton_damped(j, e, damping):
    """Just a name alias for solve_cho_damped"""
    return solve_cho_damped(A=j, b=e, damping=damping)
    
    
def solve_cho(A, b):

    if A.ndim == 2 and b.ndim == 1:
        return cho_solve(cho_factor(A), b)
    elif A.ndim == 3 and b.ndim == 2:
        nn, nx, ny = A.shape
        x = np.zeros((nn, ny))
        for i in range(nn):
            x[i] = cho_solve(cho_factor(A[i]), b[i])
        return x
    else:
        raise ValueError("solve_cho: A and b must be 2D or 3D")
        
        
def solve_cho_damped(A, b, damping):
    n, m = A.shape[-2:]
    AT = np.swapaxes(A, -2, -1)
    AAT = A @ AT

    if damping > 0:
        AAT[..., range(n), range(n)] += damping

    x = AT @ solve_cho(AAT, b)[..., np.newaxis]
    x = x[..., 0]
    return x


def matrix_sqrt(A):
    """
    Calculates the principal square root of a matrix.
    """
    if A.ndim == 1:  # A == diag
        return np.sqrt(A)

    e_val, e_vec = np.linalg.eig(A)
    if (e_val < 0).any():
        raise ValueError("Matrix does not have a real square root")
    e_val_sqrt = np.sqrt(e_val)
    sqrt_A = e_vec @ np.diag(e_val_sqrt) @ np.linalg.inv(e_vec)
    return sqrt_A


if __name__ == "__main__":
    test_dxnorm_dx()
    vis_k_farthest_neighbors()
