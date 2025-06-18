import numpy as np
from wzk.opt import wls


def fun_delta(x):
    np.random.seed(0)

    n = len(x)
    m = 100
    A = np.random.random((m, n))
    b = np.random.random(m)
    d = A @ x + b

    return d


def try_wls_jl():
    x0_list = np.random.random((100, 10))
    x_list = wls.nl_wls_mp(fun_delta=fun_delta, x0_list=x0_list, n_processes=10)
    print(x_list)
    print(x0_list)
