import numpy as np
from wzk import mp2


def random_ball_search(fun, n_outer, n_inner, x0, eps=1e-9, n_processes=1, verbose=0):
    x0 = x0.ravel()
    n_x = len(x0)

    x_outer = np.zeros((n_outer+1, n_x))
    x_outer[0] = x0
    o_outer = np.zeros(n_outer+1)
    o_outer[0] = fun(x0)

    n_termination = 20
    radius = 0.1

    def fun_loop(x_list):
        o_list = np.zeros(len(x_list))
        for l, x in enumerate(x_list):
            o_list[l] = fun(x)
        return o_list

    for i in range(n_outer):

        delta = (np.random.random((n_inner, n_x)) - 0.5) * radius

        x_inner = x_outer[i]+delta

        o_inner = mp2.mp_wrapper(x_inner, fun=fun_loop, n_processes=n_processes)

        j = np.argmin(o_inner)
        if o_inner[j] < o_outer[i]:
            o_outer[i+1] = o_inner[j]
            x_outer[i+1] = x_inner[j]
            radius = np.linalg.norm(delta[j]) * 1.5
        else:
            o_outer[i+1] = o_outer[i]
            x_outer[i+1] = x_outer[i]
            radius = radius * 0.75

        if i > n_termination and np.mean(o_outer[i-n_termination:i] - o_outer[i+1]) < eps:
            o_outer[-1] = o_outer[i]
            x_outer[-1] = x_outer[i]
            break

        if verbose > 1:
            print(f"iteration: {i} | objective: {o_outer[i+1]:.5}")

    if verbose == 1:
        print(f"iteration: {n_outer} | objective: {o_outer[-1]:.5}")

    return x_outer[-1], o_outer[-1]
