import numpy as np
from itertools import product

from . import shape as sh


def object2numeric_array(arr):
    s = np.shape(arr)
    arr = np.array([v for v in np.ravel(arr)])
    arr = np.reshape(arr, s + np.shape(arr)[1:])
    return arr


def numeric2object_array(arr):
    n = arr.shape[0]
    arr_obj = np.zeros(n, dtype=object)
    for i in range(n):
        arr_obj[i] = arr[i]

    return arr_obj


def scalar2array(*val_or_arr, shape, squeeze=True, safe=True):
    # standard numpy broadcasting rules apply
    shape = sh.shape_wrapper(shape)

    res = []
    for voa in val_or_arr:
        try:

            if isinstance(voa, str):
                dtype = np.array(voa).dtype
            elif isinstance(voa, np.ndarray):
                dtype = voa.dtype
            else:
                dtype = type(voa)

            res_i = np.empty(shape=shape, dtype=dtype)

            res_i[:] = np.array(voa).copy()
            res.append(res_i)

        except ValueError:
            if safe:
                assert np.all(np.shape(voa) == shape), f"{np.shape(voa)} != {shape}"
            res.append(voa)

    if len(res) == 1 and squeeze:
        return res[0]
    else:
        return res


def args2arrays(*args):
    return [np.array(a) for a in args]


def unify(x):
    x = np.atleast_1d(x)
    assert np.allclose(x, x.mean())
    x_mean = np.mean(x)
    return x_mean.astype(x.dtype)


def __fill_index_with(idx, axis, shape, mode="slice"):
    """
    orange <-> orth-range
    sorry but 'orange', 'slice' was just too delicious
    """
    axis = sh.axis_wrapper(axis=axis, n_dim=len(shape))
    if mode == "slice":
        idx_with_ = [slice(None) for _ in range(len(shape) - len(axis))]

    elif mode == "orange":
        idx_with_ = np.ogrid[*[range(s) for i, s in enumerate(shape) if i not in axis]]
        idx_with_ = list(idx_with_)  # TODO numpy 2.0

    elif mode is None:
        return idx
    else:
        raise ValueError(f"Unknown mode {mode}")

    idx = np.array(idx)
    for i, ax in enumerate(axis):
        idx_with_.insert(ax, idx[..., i])  # noqa

    return tuple(idx_with_)


def insert(a, val, idx, axis, mode="slice"):
    idx = __fill_index_with(idx=idx, axis=axis, shape=a.shape, mode=mode)
    a[idx] = val


def extract(a, idx, axis, mode="slice"):
    idx = __fill_index_with(idx=idx, axis=axis, shape=a.shape, mode=mode)
    return a[idx]


def __argfun(a, axis, fun):
    axis = sh.axis_wrapper(axis=axis, n_dim=a.ndim)
    if len(axis) == 1:
        return fun(a, axis=axis)

    elif len(axis) == a.ndim:
        return np.unravel_index(fun(a), shape=a.shape)

    else:
        axis_inv = sh.axis_wrapper(axis=axis, n_dim=a.ndim, invert=True)
        shape_inv = sh.get_subshape(shape=a.shape, axis=axis_inv)
        shape = sh.get_subshape(shape=a.shape, axis=axis)

        a2 = np.transpose(a, axes=np.hstack((axis_inv, axis))).reshape(shape_inv + (-1,))
        idx = fun(a2, axis=-1)
        idx = np.array(np.unravel_index(idx, shape=shape))

        return np.transpose(idx, axes=np.roll(np.arange(idx.ndim), -1))


def argmax(a, axis=None):
    return __argfun(a=a, axis=axis, fun=np.argmax)


def argmin(a, axis=None):
    return __argfun(a=a, axis=axis, fun=np.argmin)


def allclose(a, b, rtol=1.e-5, atol=1.e-8, axis=None):
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    axis = np.array(sh.axis_wrapper(axis=axis, n_dim=a.ndim))
    assert len(axis) <= len(a.shape)
    if np.isscalar(a) and np.isscalar(b):
        return np.allclose(a, b)
    shape = np.array(a.shape)[axis]
    bool_arr = np.zeros(shape, dtype=bool)
    for i in product(*(range(s) for s in shape)):
        bool_arr[i] = np.allclose(extract(a, idx=i, axis=axis),
                                  extract(b, idx=i, axis=axis),
                                  rtol=rtol, atol=atol)

    return bool_arr


def __wrapper_pair2list_fun(*args, fun):
    assert len(args) >= 2
    res = fun(args[0], args[1])
    for a in args[2:]:
        res = fun(res, a)
    return res


def minimum(*args):
    return __wrapper_pair2list_fun(*args, fun=np.minimum)


def maximum(*args):
    return __wrapper_pair2list_fun(*args, fun=np.maximum)


def logical_or(*args):
    return __wrapper_pair2list_fun(*args, fun=np.logical_or)


def logical_and(*args):
    return __wrapper_pair2list_fun(*args, fun=np.logical_and)


def max_size(*args):
    return int(np.max([np.size(a) for a in args]))


def min_size(*args):
    return int(np.min([np.size(a) for a in args]))


def argmax_size(*args):
    return int(np.argmax([np.size(a) for a in args]))


def max_len(*args):
    return int(np.max([len(a) for a in args]))


def squeeze_all(*args):
    return [np.squeeze(a) for a in args]


def round2(x,
           decimals=None):
    # noinspection PyProtectedMember
    try:
        return np.round(x, decimals=decimals)

    except (TypeError, np.core._exceptions.UFuncTypeError):  # noqa
        return np.array(x)


def clip_periodic(x, a_min, a_max):
    try:
        x = x.copy()
    except AttributeError:
        pass

    x -= a_min
    x = np.mod(x, a_max - a_min)
    x += a_min
    return x


def clip2(x, clip: float, mode: str, axis=-1):
    if mode:
        if mode == "value":
            return np.clip(x, a_min=-clip, a_max=+clip)

        elif mode == "norm":
            x = x.copy()
            n = np.linalg.norm(x, axis=axis, keepdims=True)
            b = n > clip
            x[b] = x[b] * (clip / n[b])
            return x

        elif mode == "norm-force":
            n = np.linalg.norm(x, axis=axis, keepdims=True)
            return x * (clip / n)

        else:
            raise ValueError(f"Unknown mode: '{mode}'")
    return x


def load_dict(file: str) -> dict:
    d = np.load(file, allow_pickle=True)
    try:
        d = d.item()
    except AttributeError:
        d = d["arr"]
        d = d.item()

    assert isinstance(d, dict)
    return d


def round_dict(d, decimals=None):
    for key in d.keys():
        value = d[key]
        if isinstance(value, dict):
            d[key] = round_dict(d=value, decimals=decimals)
        else:
            d[key] = round2(x=value, decimals=decimals)

    return d


def rolling_window(a, window):
    """https://stackoverflow.com/a/6811241/7570817"""
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
