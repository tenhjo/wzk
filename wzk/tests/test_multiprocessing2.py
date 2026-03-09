import time
from unittest import TestCase

import numpy as np

from wzk import ltd, mp2
from wzk.time2 import tic, toc


class Test(TestCase):
    def test_mp_wrapper(self):

        def fun__int_in(n):
            return np.zeros((n, 3))

        def fun__arr_in(arr):
            arr[:] = 0
            return arr

        def fun__multiple_arr_in(a1, a2, a3):
            a4 = a1 + a2 + a3
            return a4

        def fun__multiple_arr_out(n):
            return tuple(np.full((n, ii), ii) for ii in range(1, 5))

        def fun__multiple_out(n):
            return tuple(np.full((n, ii), ii) for ii in range(1, 3)) + (11, 12)

        n_processes = 10
        z = np.zeros((1000, 3))
        a = np.ones((1000, 3))
        b = np.ones((1000, 3)) * 2
        c = np.ones((1000, 3)) * 3

        res1 = mp2.mp_wrapper(999, fun=fun__int_in, n_processes=n_processes)
        res1b = mp2.mp_wrapper(999, fun=fun__int_in, n_processes=n_processes, use_loop=True)
        self.assertTrue(np.allclose(res1, np.zeros((999, 3))))
        self.assertTrue(ltd.list_allclose(res1, res1b))

        res2 = mp2.mp_wrapper(a, fun=fun__arr_in, n_processes=n_processes)
        res2b = mp2.mp_wrapper(a, fun=fun__arr_in, n_processes=n_processes, use_loop=True)
        self.assertTrue(np.allclose(res2, z))
        self.assertTrue(ltd.list_allclose(res2, res2b))

        res3 = mp2.mp_wrapper(a, b, c, fun=fun__multiple_arr_in, n_processes=n_processes)
        res3b = mp2.mp_wrapper(a, b, c, fun=fun__multiple_arr_in, n_processes=n_processes, use_loop=True)
        self.assertTrue(np.allclose(res3, a+b+c))
        self.assertTrue(ltd.list_allclose(res3, res3b))

        res4 = mp2.mp_wrapper(989, fun=fun__multiple_arr_out, n_processes=n_processes)
        res4b = mp2.mp_wrapper(989, fun=fun__multiple_arr_out, n_processes=n_processes, use_loop=True)
        for i in range(1, 5):
            self.assertTrue(np.allclose(res4[i-1], np.full((989, i), i)))
        self.assertTrue(all(ltd.list_allclose(res4, res4b)))

        res5 = mp2.mp_wrapper(1007, fun=fun__multiple_out, n_processes=n_processes)
        res5b = mp2.mp_wrapper(1007, fun=fun__multiple_out, n_processes=n_processes, use_loop=True)
        for i in range(1, 3):
            self.assertTrue(np.allclose(res5[i-1], np.full((1007, i), i)))
        self.assertTrue(all(ltd.list_allclose(res5, res5b)))

        res6 = mp2.mp_wrapper(1007, fun=fun__multiple_out, n_processes=n_processes, max_chunk_size=10)
        res6b = mp2.mp_wrapper(1007, fun=fun__multiple_out, n_processes=n_processes, use_loop=True)
        for i in range(1, 3):
            self.assertTrue(np.allclose(res6[i-1], np.full((1007, i), i)))
        self.assertTrue(all(ltd.list_allclose(res6[:2], res6b[:2])))

    def test_time(self):

        def fun_time():
            time.sleep(1)
            return np.ones(1)

        n_processes = 100
        tic()
        _ = mp2.mp_wrapper(fun=fun_time, n_processes=n_processes)
        toc()


if __name__ == "__main__":
    pass
    test = Test()
    test.test_time()
