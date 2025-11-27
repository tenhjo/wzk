from time import sleep
import numpy as np
import multiprocessing
from wzk import ltd

# Error under Python3.8 /macOS -> AttributeError: Can't pickle a local object 'mp_wrapper.<locals>.__fun_wrapper'
# https://stackoverflow.com/questions/60518386/error-with-module-multiprocessing-under-python3-8
try:
    multiprocessing.set_start_method("fork", force=True)
except ValueError:
    multiprocessing.set_start_method("spawn", force=True)  # use fallback for other OS


def n_processes_wrapper(n_processes, n_samples):
    return min(max(1, n_processes), n_samples)


def get_n_samples_per_process(n_samples, n_processes):
    n_samples_per_core = n_samples // n_processes
    odd_idx = n_samples - (n_samples_per_core * n_processes)
    n_samples_per_core = np.repeat(n_samples_per_core, repeats=n_processes)
    n_samples_per_core[:odd_idx] += 1

    n_samples_per_core_cs = np.cumsum(np.concatenate([[0], n_samples_per_core]))

    return n_samples_per_core, n_samples_per_core_cs


def vectorize(fun, *args, n_dimx=1):

    def fun_vec(x_vec):
        if np.ndim(x_vec) == n_dimx:
            return fun(x_vec)
        elif np.ndim(x_vec) == n_dimx + 1:
            return np.array([fun(x, *args) for x in x_vec])
        else:
            raise ValueError

    return fun_vec


def mp_wrapper(*args, fun,
               n_processes=1, max_chunk_size=None, use_loop=False,
               start_method=None):
    """
    Multiprocessing Wrapper for a function with a single argument.
    arg must be an iterative and will be split along its first dimension and fed to the different processes
    The objects returned from fun must be numpy arrays and also be splittable / combinable along the first dimension

    if this is not the case, write a wrapper function so that fun expects no arguments

    Caveats:
    - The function breaks if the data passed through the pipe is too large,
      so make sure that the data size does not exceed ~100Mb per process
      https://stackoverflow.com/questions/31552716/multiprocessing-queue-full

    - Numpy's random number generator starts with the same seed for each process, so if your calculations depend on
      those random values, all processes will work with the same numbers. The default behavior is to call
      np.random.seed() in each process to make sure the random numbers of each process are different. If you want
      them to be the same, use np.random.seed(42) in the function directly
      https://stackoverflow.com/a/12915206/7570817

    - Be careful with functions that only return a value whose shape does not depend on the input,
      In such cases different number of processes leads to different results; especially the keyword max_chunk_size
      might cause trouble

    - max_chunk_size is intended for cases where both a loop and parallelization is needed because of memory limitations
    """

    if start_method is not None:
        multiprocessing.set_start_method(start_method, force=True)
    time_sleep = 0.01  # s

    if len(args) == 0:
        n_samples = n_processes
    elif len(args) == 1 and isinstance(args[0], int):
        n_samples = args[0]
        # args = []
    else:
        n_samples = len((args[0]))

    n_processes = n_processes_wrapper(n_processes=n_processes, n_samples=n_samples)

    if n_processes == 1:
        return fun(*args)

    if len(args) == 0:
        def fun_i():
            np.random.seed(None)
            return fun()

        def __fun_wrapper(i_process, queue):
            queue.put((i_process, fun_i()))

    else:
        n_samples_pp, n_samples_pp_cs = \
            get_n_samples_per_process(n_samples=n_samples, n_processes=n_processes)

        if isinstance(args[0], int):
            if max_chunk_size is None:
                def fun_i(i_process):
                    np.random.seed(None)

                    return fun(n_samples_pp[i_process])

            else:
                def fun_i(i_process):
                    np.random.seed(None)

                    n_s = n_samples_pp[i_process]
                    ns_pp, ns_pp_cs = get_n_samples_per_process(n_samples=n_s, n_processes=n_s // max_chunk_size)
                    return combine_results([fun(ns_pp_i) for ns_pp_i in ns_pp])

        else:
            if max_chunk_size is not None:
                def fun_i(i_process):
                    np.random.seed(None)

                    n_s = n_samples_pp[i_process]
                    ns_pp, ns_pp_cs = get_n_samples_per_process(n_samples=n_s, n_processes=n_s // max_chunk_size)
                    return combine_results([fun(*map(lambda a: a[ns_pp_cs[j]:ns_pp_cs[j+1]], args))
                                            for j in range(len(ns_pp_cs)-1)])
            else:
                def fun_i(i_process):
                    np.random.seed(None)

                    return fun(*map(lambda a: a[n_samples_pp_cs[i_process]:n_samples_pp_cs[i_process+1]],
                                    args))

        if use_loop:
            return combine_results([fun_i(i_process=ip) for ip in range(n_processes)])

        def __fun_wrapper(i_process, queue):
            queue.put((i_process, fun_i(i_process=i_process)))

    # Start the processes and save their results in a queue
    result_queue = multiprocessing.Queue(n_processes)
    process_list = []
    for i in range(n_processes):
        p = multiprocessing.Process(target=__fun_wrapper, args=(i, result_queue), name=str(i))
        p.start()
        process_list.append(p)

    # Make sure the queue is full before joining the processes with a small timeout, otherwise there occurred
    #  errors because a process filled the queue but was still alive
    while True:
        sleep(time_sleep)
        if result_queue.full():
            break

    # Wait for the processes to finish
    for i in range(n_processes):
        process_list[i].join(timeout=time_sleep)

    # Combine and return the results
    results = [result_queue.get() for _ in range(n_processes)]

    # - order the results according to the process indices
    results = ltd.change_tuple_order(results)
    idx = np.argsort(results[0])
    results = [results[1][i] for i in idx]

    return combine_results(results=results)


def combine_results(results):
    if isinstance(results[0], tuple):
        results = ltd.change_tuple_order(results)
        return tuple([np.concatenate(r, axis=0) if np.ndim(r[0]) > 0 else np.array(r)
                      for r in results])
    else:
        if results[0] is None:
            return None
        else:
            return np.concatenate(results, axis=0)
