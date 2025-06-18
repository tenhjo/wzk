import numpy as np
from wzk import printing, math2, mp2, mpl2


def idx_times_all(idx, n):
    idx = np.atleast_2d(idx)
    idx2 = idx.repeat(n, axis=0)
    idx2 = np.hstack((idx2, np.tile(np.arange(n), reps=idx.shape[0])[:, np.newaxis])).astype(int)

    return idx2


def greedy(n, k, fun, i0=None, verbose=0):
    """
    choose k elements out of set with size n
    fun(idx_list) measures how good the current choice is

    """

    if i0 is not None:
        s = np.array(i0).tolist()
    else:
        s = []

    for i in range(k):
        printing.progress_bar(i=i, n=k, eta=True, verbose=verbose)
        idx_i = idx_times_all(idx=s, n=n)
        o = fun(idx_i)
        o[s] = np.inf
        s.append(np.argmin(o))

    # best = np.sort(best)
    s = np.array(s, dtype=int)
    o = fun(s)
    if verbose > 1:
        print(f"set: {repr(s)} | objective: {o}")
    return s, o


def detmax(fun, x0=None, n=100, k=30, excursion=10, method="add->remove", max_loop=3,
           verbose=0):
    """
    method:  'add->remove'
             'remove->add'
    """

    improvement_threshold = 1e-2
    if x0 is None:
        x0 = greedy(n=n, k=k, fun=fun, verbose=verbose-1)
        # x0 = math2.random_subset(n=n, k=k, m=1, dtype=np.int16)[0]

    def __add(x, nn):
        x = idx_times_all(idx=x, n=nn)
        oo = fun(x)
        oo[x[0, :-1]] = np.inf
        idx_min = np.argmin(oo)
        # idx_min = np.random.choice(np.argsort(oo)[:nn//10])
        oo = oo[idx_min]
        x = x[idx_min]
        return x, oo

    def remove(x, exc):
        oo = None
        for _ in range(1, exc+1):
            x = np.repeat([x], repeats=len(x), axis=0)
            x = x[np.logical_not(np.eye(len(x), dtype=bool))].reshape(len(x), len(x)-1)

            oo = fun(x)
            idx_min = np.argmin(oo)
            # idx_min = np.random.choice(np.argsort(oo)[:100//10])

            oo = oo[idx_min]
            x = x[idx_min]

        return np.sort(x), oo

    def add(x, nn, exc):
        oo = None
        for _ in range(1, exc+1):
            x, oo = __add(x=x, nn=nn)

        return np.sort(x), oo

    def addremove(x, nn, exc):  # noqa
        x = np.repeat([x], repeats=len(x), axis=0)
        x = x[np.logical_not(np.eye(len(x), dtype=bool))].reshape(len(x), len(x) - 1)

        x, oo = __add(x=x, nn=nn)
        return np.sort(x), oo

    o = np.inf
    for q in range(1, excursion+1):

        i = 0
        for i in range(max_loop):
            o_old = o
            if method == "add->remove":
                x0, o = add(x=x0, nn=n, exc=q)
                x0, o = remove(x=x0, exc=q)
            elif method == "remove->add":
                x0, o = remove(x=x0, exc=q)
                x0, o = add(x=x0, nn=n, exc=q)
            elif method == "both":
                raise NotImplementedError()
            else:
                raise ValueError("Unknown method, see doc string for more information")

            if o_old - o < improvement_threshold:
                break

        if verbose >= 2:
            print("Depth: {} | Loop {} | Objective: {:.4} | Configuration: {} ".format(q, i+1, o, x0))

    if verbose >= 1:
        print(" Objective: {:.4} | Configuration: {}".format(o, x0))
    return x0, o


def random(n, k, m, fun, chunk=1000,
           n_processes=10,
           dtype=np.uint8, verbose=0):

    def fun2(_m):
        _idx = math2.random_subset(n=n, k=k, m=_m, dtype=dtype)
        _o = fun(_idx)
        return _idx, _o

    idx, o = mp2.mp_wrapper(m, fun=fun2, n_processes=n_processes, max_chunk_size=chunk)

    if verbose > 1:
        fig, ax = mpl2.new_fig()
        ax.hist(o, bins=100)

    i_sorted = np.argsort(o)
    o = o[i_sorted]
    idx = idx[i_sorted].astype(int)

    return idx, o


def ga(n, k, m, fun, verbose, **kwargs):
    from wzk.ga.kofn import kofn

    best, ancestors = kofn(n=n, k=k, fitness_fun=fun,  pop_size=m, verbose=verbose, **kwargs)

    print(repr(best))
    return best, ancestors
