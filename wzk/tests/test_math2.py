from unittest import TestCase

import numpy as np
from wzk import math2


class Test(TestCase):

    def test_normalize_01(self):
        x = np.arange(20).reshape(4, 5)
        x[-1, -1] = 20
        sol = np.array([[0., 0.05, 0.1, 0.15, 0.2],
                        [0.25, 0.3, 0.35, 0.4, 0.45],
                        [0.5, 0.55, 0.6, 0.65, 0.7],
                        [0.75, 0.8, 0.85, 0.9, 1.]])
        res = math2.normalize_01(x)
        self.assertTrue(np.allclose(sol, res))

        x = np.arange(40).reshape((2, 4, 5))
        x[0, -1, -1] = 20
        x[1, -1, -1] = 40
        res = math2.normalize_01(x, axis=(-2, -1))
        self.assertTrue(np.allclose(sol, res[0]))
        self.assertTrue(np.allclose(sol, res[1]))

    def test_divisor(self):
        sol = [[], [], [], [], [2], [],
               [2, 3], [], [2, 4], [3], [2, 5],
               [], [2, 4, 3, 6], [], [2, 7], [3, 5],
               [2, 4, 8], [], [2, 3, 6, 9], [], [2, 4, 5, 10],
               [3, 7], [2, 11], [], [2, 4, 8, 3, 6, 12], [5],
               [2, 13], [3, 9], [2, 4, 7, 14], [], [2, 3, 6, 5, 10, 15],
               [], [2, 4, 8, 16], [3, 11], [2, 17], [5, 7],
               [2, 4, 3, 6, 12, 9, 18], [], [2, 19], [3, 13], [2, 4, 8, 5, 10, 20],
               [], [2, 3, 6, 7, 14, 21], [], [2, 4, 11, 22], [3, 9, 5, 15],
               [2, 23], [], [2, 4, 8, 16, 3, 6, 12, 24], [7], [2, 5, 10, 25],
               [3, 17], [2, 4, 13, 26], [], [2, 3, 6, 9, 18, 27], [5, 11],
               [2, 4, 8, 7, 14, 28], [3, 19], [2, 29], [], [2, 4, 3, 6, 12, 5, 10, 20, 15, 30],
               [], [2, 31], [3, 9, 7, 21], [2, 4, 8, 16, 32], [5, 13],
               [2, 3, 6, 11, 22, 33], [], [2, 4, 17, 34], [3, 23], [2, 5, 10, 7, 14, 35],
               [], [2, 4, 8, 3, 6, 12, 24, 9, 18, 36], [], [2, 37], [3, 5, 15, 25],
               [2, 4, 19, 38], [7, 11], [2, 3, 6, 13, 26, 39], [], [2, 4, 8, 16, 5, 10, 20, 40],
               [3, 9, 27], [2, 41], [], [2, 4, 3, 6, 12, 7, 14, 28, 21, 42], [5, 17],
               [2, 43], [3, 29], [2, 4, 8, 11, 22, 44], [], [2, 3, 6, 9, 18, 5, 10, 15, 30, 45],
               [7, 13], [2, 4, 23, 46], [3, 31], [2, 47], [5, 19],
               [2, 4, 8, 16, 32, 3, 6, 12, 24, 48], [], [2, 7, 14, 49], [3, 9, 11, 33], [2, 4, 5, 10, 20, 25, 50]]
        res = [math2.divisors(i) for i in range(101)]
        self.assertEqual(sol, res)

    def test_mean_divisor_pair(self):
        self.assertEqual((1, 1), math2.get_mean_divisor_pair(1))
        self.assertEqual((1, 2), math2.get_mean_divisor_pair(2))
        self.assertEqual((2, 5), math2.get_mean_divisor_pair(10))
        self.assertEqual((3, 4), math2.get_mean_divisor_pair(12))
        self.assertEqual((4, 6), math2.get_mean_divisor_pair(24))
        self.assertEqual((9, 9), math2.get_mean_divisor_pair(81))
        self.assertEqual((9, 11), math2.get_mean_divisor_pair(99))

    def test_discretize(self):
        sol = np.array([0., 0.17, 0.17, 0.34, 0.34, 0.51, 0.68, 0.68, 0.85, 1.02,
                        1.02, 1.19, 1.19, 1.36, 1.53, 1.53, 1.70, 1.87, 1.87, 2.04])

        res = math2.discretize(x=np.linspace(0, 2, num=20), step=0.17)
        self.assertTrue(np.allclose(sol, res))

    def test_d_linalg_norm__d_x(self):

        n = 1000
        n_dof = 10
        x = np.arange(1, n + 1)[:, np.newaxis] * np.random.random((n, n_dof))

        def linalg_norm(q):
            return q / np.linalg.norm(q, axis=-1, keepdims=True)

        def ln2(q):
            q = linalg_norm(q)
            q = (q * np.arange(1, q.shape[-1] + 1)).sum(axis=-1)
            return q

        jac_num = math2.numeric_derivative(fun=linalg_norm, x=x, axis=-1)
        jac_num2 = math2.numeric_derivative(fun=ln2, x=x, axis=-1)
        jac = math2.dxnorm_dx(x)

        jac2 = (jac @ np.arange(1, x.shape[-1] + 1)[:, np.newaxis])[..., 0]

        self.assertTrue(np.allclose(jac_num, jac))
        self.assertTrue(np.allclose(jac_num2, jac2))

    def test_generative_derv(self):
        def x54321(x):
            return (x ** 5 + x ** 4 + x ** 3 + x ** 2 + x)[..., 0]

        def x54321_derv(x):
            return 5 * x ** 4 + 4 * x ** 3 + 3 * x ** 2 + 2 * x + 1

        a = np.random.random((100, 10, 1))
        grad_analytic = x54321_derv(x=a)
        grad_numeric = math2.numeric_derivative(fun=x54321, x=a, axis=-1)

        self.assertTrue(np.allclose(grad_analytic, grad_numeric))

    def test_matrix_sqrt(self):
        A = np.array([[1, 2], [2, 4]])
        sqrt_A = math2.matrix_sqrt(A)
        self.assertTrue(np.allclose(sqrt_A @ sqrt_A, A))
