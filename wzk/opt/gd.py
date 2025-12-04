import numpy as np  # noqa

from wzk import np2, mp2, time2, ltd, strings, files

from wzk.opt.optimizer import Naive


__DEBUG = False
__debug_directory = "/Users/jote/Documents/PhD/data/mopla/GD/debug"


class OPTimizer(ltd.CopyableObject):
    __slots__ = ("type",                           # str                 | type of the optimizer: gd, sqp, ...
                 "type_root",                      # str                 | type of the root search algorithm

                 "n_samples",                      # int                 | Number of samples

                 "n_steps",                        # int                 | Number of iterations
                 "stepsize",                       # float               |

                 "optimizer",                      # Optimizer           | Adam, RMSProp, ...

                 "limits",                         # fun()               | Limits of the variables
                 "limits_mode",                    # str                 | "jump", "clip", "ignore"

                 "clip",                           # float[n_steps]      |
                 "clip_mode",                      # str                 | "value", "norm", "norm-force" (see np2.clip2)


                 "callback",                       # fun()               |
                 "n_processes",                    # int                 |
                 "use_loop_instead_of_processes",  # bool                |
                 "hesse_inv",                      # float[n_var][n_var] |
                 "hesse_weighting",                # float[n_steps]      |
                 "active_dims",                    # bool[n_var]         |
                 "staircase",                      # OPTStaircase        |
                 "return_x_list",                  # bool                | s this a suitable parameter? not really
                 "verbose"                         # int
                 )

    def __init__(self, n_samples=1, n_steps=100, stepsize=1, optimizer=Naive(), clip=0.1, n_processes=1,
                 clip_mode="value", limits_mode="clip"):

        self.type = None
        self.type_root = "newton"

        self.n_samples = n_samples

        self.n_steps = n_steps
        self.stepsize = stepsize

        self.clip = clip
        self.clip_mode = clip_mode
        self.limits_mode = limits_mode
        self.limits = lambda x: x

        self.optimizer = optimizer
        self.n_processes = n_processes
        self.active_dims = None
        self.use_loop_instead_of_processes = False
        self.staircase = OPTStaircase(n_stairs=-1)

        self.callback = None

        self.hesse_inv = None
        self.hesse_weighting = 0

        # Logging
        self.return_x_list = False

        self.verbose = 0


class OPTStaircase(object):
    __slots__ = ("n_stairs",        # int
                 "n_var",           # int[n_stairs]
                 "n_steps",         # int[n_stairs]
                 "clip",            # float[n_stairs]
                 "hesse_inv_dict",  # dict[n_stairs]
                 )

    def __init__(self, n_stairs=-1):
        self.n_stairs = n_stairs


# Gradient Descent
def gradient_descent_mp(x, fun, grad, opt):

    def gd_wrapper(xx):
        return gradient_descent(x=xx, fun=fun, grad=grad, opt=opt)

    return mp2.mp_wrapper(x, fun=gd_wrapper, n_processes=opt.n_processes)


def gradient_descent(x, fun, grad, opt):

    x = __x_wrapper(x)

    if opt.active_dims is not None:
        active_dims = opt.active_dims
    else:
        active_dims = slice(x.shape[-1])

    # If the parameters aren't given for all steps, expand them
    if np.size(opt.clip) == 1:
        opt.clip = np.full(opt.n_steps, fill_value=float(opt.clip))

    if np.size(opt.hesse_weighting) == 1:
        opt.hesse_weighting = np.full(opt.n_steps, fill_value=float(opt.hesse_weighting))

    # grad_max_evolution = []
    if opt.return_x_list or __DEBUG:
        x_list = np.zeros((opt.n_steps,) + x.shape)
        o_list = np.zeros((opt.n_steps, len(x)))
    else:
        x_list = None
        o_list = None

    # Gradient Descent Loop
    for i in range(opt.n_steps):
        j = grad(x=x, i=i)

        # Correct via an approximated hesse function
        if opt.hesse_inv is not None and opt.hesse_weighting[i] > 0:
            h = (opt.hesse_inv[np.newaxis, ...] @ j.reshape(-1, opt.hesse_inv.shape[-1], 1)).reshape(j.shape)
            j = j * (1 - opt.hesse_weighting[i]) + h * opt.hesse_weighting[i]

        if opt.callback is not None:
            j = opt.callback(x=x.copy(), jac=j.copy())  # , count=o) -> callback function handles count

        v = opt.optimizer.update(x=x, v=j)
        v = np2.clip2(v, clip=opt.clip[i], mode=opt.clip_mode)
        x[..., active_dims] += v[..., active_dims]

        x = opt.limits(x)

        if opt.return_x_list or __DEBUG:
            x_list[i] = x
            o_list[i] = fun(x)  # only for debugging, is inefficient to call separately

        if opt.verbose > 0:
            print(i, fun(x))

    o = fun(x)

    if __DEBUG:
        file = f"{__debug_directory}/gd_{time2.get_timestamp(millisecond=True)}__{strings.uuid4()}"
        files.save_pickle(file=file, obj=dict(x=x_list, o=o_list))

    if opt.return_x_list:
        return x, o, (x_list, o_list)
    else:

        return x, o


def __x_wrapper(x):
    x = np.array(x.copy())
    if x.ndim == 1:
        x = x[np.newaxis, :]

    return x
