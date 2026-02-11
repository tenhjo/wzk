
# import numpy as np
# import time

# from wzk.ray2 import ray, init
# from wzk import tic, toc, mp2
#
#
# def speed():
#     # nodes = ['rmc-lx0062', 'philotes', 'polyxo', 'poros',
#     #          'rmc-galene', 'rmc-lx0271', 'rmc-lx0141', 'rmc-lx0392']
#     # n_cpu = start_ray_cluster(head=None, nodes=nodes, perc=50)
#     n_cpu = 30
#     init()
#
#     # ray.init(address='auto', log_to_driver=False)
#
#     n = n_cpu
#     t = 1
#
#
#     def test0(*args):   # noqa
#         log_print("a")
#         time.sleep(t)
#         return np.ones((2, 1))
#
#     @ray.remote
#     def test():
#         time.sleep(t)
#         return 1
#
#     def test_ray():
#         tic()
#         futures = []
#         for _ in range(n):
#             futures.append(test.remote())
#         _ = ray.get(futures)
#         tt = toc("ray")
#         return tt
#
#     # def compare_times(tt):
#     #     t0 = (t*n/n_cpu)
#     #     log_print(t0)
#     #     log_print((tt - t0)/n)
#
#     test_ray()
#     test_ray()
#     test_ray()
#     # 0.0008937692642211914
#     # 0.001179349422454834
#
#     tic()
#     for i in range(n//n_cpu):
#         test0()
#     toc()
#
#     tic()
#     mp2.mp_wrapper(n, fun=test0, n_processes=n_cpu)
#     toc("mp")
