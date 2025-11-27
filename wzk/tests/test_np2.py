from unittest import TestCase

from typing import Literal
import numpy as np
from wzk import np2, testing


class Test(TestCase):

    def test_DummyArray(self):
        arr = np.random.random((4, 4))
        idx = (1, 2, 3, 4)
        d = np2.DummyArray(arr=arr, shape=(4, 5, 6, 6))
        self.assertTrue(np.allclose(arr, d[idx]))

        d = np2.DummyArray(arr=1, shape=(2, 2))
        self.assertTrue(np.allclose(1, d[1, :]))

    def test_initialize_array(self):

        shape = [4, (4,), (1, 2, 3, 4)]
        dtype = [float, int, bool]
        order = ["C", "F"]

        for s in shape:
            for d in dtype:
                for o in order:
                    o: Literal["C", "F"]

                    self.assertTrue(testing.compare_arrays(a=np2.initialize_array(shape=s, dtype=d, order=o,
                                                                                  mode="zeros"),
                                                           b=np.zeros(shape=s, dtype=d, order=o)))

                    self.assertTrue(testing.compare_arrays(a=np2.initialize_array(shape=s, dtype=d, order=o,
                                                                                  mode="ones"),
                                                           b=np.ones(shape=s, dtype=d, order=o)))

                    self.assertTrue(testing.compare_arrays(a=np2.initialize_array(shape=s, dtype=d, order=o,
                                                                                  mode="empty"),
                                                           b=np.empty(shape=s, dtype=d, order=o)))

                    np.random.seed(0)
                    a = np2.initialize_array(shape=s, dtype=d, order=o, mode="random")
                    np.random.seed(0)
                    b = np.random.random(s).astype(dtype=d, order=o)
                    self.assertTrue(testing.compare_arrays(a=a, b=b))

    def test_np_isinstance(self):

        self.assertTrue(np2.np_isinstance(4.4, float))
        self.assertFalse(np2.np_isinstance(4.4, int))

        self.assertTrue(np2.np_isinstance(("this", "that"), tuple))
        self.assertTrue(np2.np_isinstance(("this", "that"), tuple))

        self.assertTrue(np2.np_isinstance(np.full((4, 4), "bert"), str))
        self.assertTrue(np2.np_isinstance(np.ones((5, 5), dtype=bool), bool))

        self.assertTrue(np2.np_isinstance(np.ones(4, dtype=int), int))
        self.assertFalse(np2.np_isinstance(np.ones(4, dtype=int), float))

    def test_insert(self):
        a = np.ones((4, 5, 3))
        val = 2

        np2.insert(a=a, idx=(1, 2), axis=(0, 2), val=val)

        self.assertTrue(np.allclose(a[1, :, 2], val))

    def test_argmax(self):
        n = 100
        axis = (0, 2)
        size = (3, 4, 5, 6)

        a = np.random.randint(n, size=size)
        i = np2.argmax(a, axis=axis)

        e = np2.extract(a=a, axis=axis, idx=i, mode="orange")
        # e = extract(a=a, axis=axis, idx=i, mode='slice')
        amax = np.max(a, axis=axis)
        self.assertTrue(np.allclose(amax, e))

    def test_argmin(self):
        n = 1000
        axis = (1, 3, 5)
        size = (3, 4, 5, 6, 7, 8, 9)

        a = np.random.randint(n, size=size)
        i = np2.argmin(a, axis=axis)

        e = np2.extract(a=a, axis=axis, idx=i, mode="orange")
        amin = np.min(a, axis=axis)
        self.assertTrue(np.allclose(amin, e))

    def test_safe2vectors(self):
        self.assertTrue(np.array_equal([np.array([1])], np2.scalar2array(1, shape=1, squeeze=False)))
        self.assertTrue(np.array_equal(np.array([1]), np2.scalar2array(1, shape=1, squeeze=True)))

        self.assertTrue(np.array_equal([np.array(["a", "a", "a"], dtype="<U1"),
                                        np.array(["b", "b", "b"], dtype="<U1"),
                                        np.array(["c", "c", "c"], dtype="<U1")],
                                       np2.scalar2array("a", "b", "c", shape=3)))

        self.assertTrue(np.array_equal([np.array([1, 1, 1]),
                                        np.array([None, None, None], dtype=object),
                                        np.array(["a", "a", "a"], dtype="<U1")],
                                       np2.scalar2array(1, None, "a", shape=3)))

    def test_find_values(self):
        arr = np.array([3, 5, 5, 6, 7, 8, 8, 8, 10, 11, 1])
        values = [3, 5, 8]
        res = np2.find_values(arr=arr, values=values)
        true = np.array([True, True, True, False, False, True, True, True, False, False, False])

        self.assertTrue(np.array_equal(res, true))

    def test_tile_offset(self):
        a = np.arange(3)
        res = np2.tile_offset(a=a, reps=3, offsets=10)
        true = np.array([0, 1, 2, 10, 11, 12, 20, 21, 22])
        self.assertTrue(np.array_equal(res, true))

        a = np.arange(12).reshape(3, 4)
        res = np2.tile_offset(a=a, reps=2, offsets=(100, 1000))
        true = np.array([[0, 1, 2, 3, 1000, 1001, 1002, 1003],
                         [4, 5, 6, 7, 1004, 1005, 1006, 1007],
                         [8, 9, 10, 11, 1008, 1009, 1010, 1011]])
        self.assertTrue(np.array_equal(res, true))

        a = np.arange(12).reshape(4, 3)
        res = np2.tile_offset(a=a, reps=(2, 3), offsets=(100, 1000))
        true = np.array([[0, 1, 2, 1000, 1001, 1002, 2000, 2001, 2002],
                         [3, 4, 5, 1003, 1004, 1005, 2003, 2004, 2005],
                         [6, 7, 8, 1006, 1007, 1008, 2006, 2007, 2008],
                         [9, 10, 11, 1009, 1010, 1011, 2009, 2010, 2011],
                         [100, 101, 102, 1100, 1101, 1102, 2100, 2101, 2102],
                         [103, 104, 105, 1103, 1104, 1105, 2103, 2104, 2105],
                         [106, 107, 108, 1106, 1107, 1108, 2106, 2107, 2108],
                         [109, 110, 111, 1109, 1110, 1111, 2109, 2110, 2111]])
        self.assertTrue(np.array_equal(res, true))

    def test_construct_array(self):
        b = np2.construct_array(shape=10, val=[1, 2, 3], idx=[2, 4, 5], dtype=None, insert_mode=None)
        self.assertTrue(np.allclose(b, [0, 0, 1, 0, 2, 3, 0, 0, 0, 0]))

    def test_round_dict(self):
        d = {"a": 1.123,
             "b": {"c": np.arange(5) / 27,
                   "d": {"e": "why",
                         "f": ["why", "not", "why"]
                         }
                   }
             }

        d_round = np2.round_dict(d=d, decimals=1)
        print(d_round)

    def test_get_interval_indices(self):
        arr, res = [0]*7, [0]*7
        arr[0], res[0] = np.array([0, 0, 0, 0]), np.zeros((0, 2))
        arr[1], res[1] = np.array([0, 0, 0, 1]), np.array([[3, 4]])
        arr[2], res[2] = np.array([0, 1, 1, 0]), np.array([[1, 3]])
        arr[3], res[3] = np.array([1, 0, 0, 0]), np.array([[0, 1]])
        arr[4], res[4] = np.array([1, 0, 0, 1]), np.array([[0, 1], [3, 4]])
        arr[5], res[5] = np.array([1, 1, 0, 1]), np.array([[0, 2], [3, 4]])
        arr[6], res[6] = np.array([1, 1, 1, 1]), np.array([[0, 4]])

        for aa, rr in zip(arr, res):
            self.assertTrue(np.array_equal(np2.get_interval_indices(aa), rr))

    def try_clip_periodic(self, verbose=0):
        a_min = 5
        a_max = 37

        x = np.linspace(start=0, stop=100, num=10000)
        x2 = np2.clip_periodic(x=x, a_min=a_min, a_max=a_max)

        self.assertTrue(np.all(x2 >= a_min))
        self.assertTrue(np.all(x2 <= a_max))

        if verbose > 10:
            from wzk import mpl2
            fig, ax = mpl2.new_fig()
            ax.plot(x, x2, color="blue")
            ax.hlines(y=[a_min, a_max], xmin=np.min(x), xmax=np.max(x), color="red")

    def test_diag_wrapper(self):
        a = np.array([[2, 0, 0, 0],
                      [0, 2, 0, 0],
                      [0, 0, 2, 0],
                      [0, 0, 0, 2]])
        b = np.array([[2, 0, 0],
                      [0, 3, 0],
                      [0, 0, 4]])

        self.assertTrue(np.allclose(np2.diag_wrapper(n=4, x=2), a))
        self.assertTrue(np.allclose(np2.diag_wrapper(n=3, x=[2, 3, 4]), b))

        self.assertTrue(np.allclose(np2.diag_wrapper(n=4, x=a), a))
        self.assertTrue(np.allclose(np2.diag_wrapper(n=3, x=b), b))

    def test_slicen(self):
        sl = np2.slicen([1], [2])
        self.assertTrue(sl == (slice(1, 2),))


if __name__ == "__main__":
    pass
