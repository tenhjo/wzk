import unittest
from itertools import combinations
import numpy as np
from wzk import geometry, testing, printing


class Test(unittest.TestCase):

    def test_rotation_between_vectors(self):
        a, b = np.random.random((2, 3))
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        r_ab = geometry.rotation_between_vectors(a=a, b=b)
        r_ba = geometry.rotation_between_vectors(a=b, b=a)

        b1 = (r_ab @ a[:, np.newaxis])[:, 0]
        a1 = (r_ba @ b[:, np.newaxis])[:, 0]

        self.assertTrue(testing.compare_arrays(a, a1))
        self.assertTrue(testing.compare_arrays(b, b1))
        self.assertTrue(testing.compare_arrays(r_ab, r_ba.T))

    def test_get_orthonormal(self):
        a = np.random.random(3)
        b = geometry.get_orthonormal(a)
        self.assertTrue(np.allclose(np.dot(a, b), 0))

    def speed_mink(self):
        from wzk import tic, toc
        n = 12
        m = 50
        lines = np.random.random((m, n, 2, 3))
        pairs = np.array(list(combinations(np.arange(n), 2)))

        tic()
        n = 40
        _ = np.random.random((m, n, 3))
        _ = np.array(list(combinations(np.arange(n), 2)))

        tic()
        for i in range(100):
            _ = geometry.line_line_pairs(lines=lines, pairs=pairs)
        toc("mink")

        tic()
        for i in range(100):
            _ = geometry.line_line_pairs(lines=lines, pairs=pairs)
        toc("d1234")

        self.assertTrue(True)

    def __check_capsule_capsule(self, capsule_a, capsule_b, radius_a, radius_b, d_true=None):

        xa, xb, d00 = geometry.capsule_capsule(line_a=capsule_a[::+1], radius_a=radius_a,
                                               line_b=capsule_b[::+1], radius_b=radius_b)
        xa, xb, d01 = geometry.capsule_capsule(line_a=capsule_a[::+1], radius_a=radius_a,
                                               line_b=capsule_b[::-1], radius_b=radius_b)
        xa, xb, d10 = geometry.capsule_capsule(line_a=capsule_a[::-1], radius_a=radius_a,
                                               line_b=capsule_b[::+1], radius_b=radius_b)
        xa, xb, d11 = geometry.capsule_capsule(line_a=capsule_a[::-1], radius_a=radius_a,
                                               line_b=capsule_b[::-1], radius_b=radius_b)

        d = np.array([d00, d01, d10, d11])
        self.assertTrue(np.allclose(d, d.mean()), msg=f"{d.mean()} | {d}")

        if d_true is not None:
            self.assertTrue(np.allclose(d, d_true), msg=f"{d_true} | {d.mean()} | {d}")

    def test_capsule_capsule_permutations(self):
        m = 1000
        for i in range(m):
            printing.progress_bar(prefix="capsule_capsule_permutation", i=i, n=m, eta=True)

            capsule_a, capsule_b = np.random.random((2, 2, 3))
            radius_a, radius_b = np.random.random(2)

            self.__check_capsule_capsule(capsule_a=capsule_a, capsule_b=capsule_b,
                                         radius_a=radius_a, radius_b=radius_b)

    def test_capsule_capsule_closest(self):

        offset = 0.01
        capsule_a = np.array([[-offset, 0.0, 0.0],
                              [-1, -1, -1]])
        capsule_b = np.array([[+offset, 0.0, 0.0],
                              [-1, -1, -1]])

        m = 1000
        for i in range(m):
            printing.progress_bar(prefix="capsule_capsule_closest", i=i, n=m, eta=True)
            capsule_a[1] = np.random.random(3)
            capsule_a[1, 0] -= 1 + offset
            capsule_b[1] = np.random.random(3)
            capsule_b[1, 0] += offset

            radius_a, radius_b = np.random.uniform(low=0, high=offset / 2, size=2)

            d_true = 2 * offset - radius_a - radius_b
            self.__check_capsule_capsule(capsule_a=capsule_a, capsule_b=capsule_b,
                                         radius_a=radius_a, radius_b=radius_b, d_true=d_true)
