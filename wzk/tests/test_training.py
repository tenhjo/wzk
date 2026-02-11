
from unittest import TestCase
import numpy as np
from wzk import training


class Test(TestCase):

    def test_n2train_test(self):
        self.assertTrue(training.n2train_test(n=1000, split=0.2), (200, 800))
        self.assertTrue(training.n2train_test(n=1000, split=0), (0, 1000))
        self.assertTrue(training.n2train_test(n=1000, split=1), (1000, 0))
        self.assertTrue(training.n2train_test(n=1000, split=10), (10, 990))
        self.assertRaises(ValueError, training.n2train_test, 1000, 1.1)

    def __assert(self, res, s, n):
        for x_train, x_test in zip(*res):
            # log_print(x_train, x_test)
            self.assertTrue(len(x_test) == s)
            self.assertTrue(len(x_test) + len(x_train) == n or
                            len(x_test) + len(x_train) == 2*n)

    def test_train_test_split(self):

        n = 10
        split = 4
        a = np.ones((n, 1))
        b = np.ones((n, 2)) * 2
        c = np.ones((n, 3)) * 3
        train_test_tuple = training.train_test_split(a, b, c, split=split)
        self.__assert(train_test_tuple, s=split, n=n)

        n = 100
        split = 0.2
        a = np.ones((n, 10, 1))
        b = np.ones((n, 20, 2)) * 2
        c = np.ones((n, 30, 3)) * 3
        train_test_tuple = training.train_test_split(a, b, c, split=split)
        self.__assert(train_test_tuple, s=split*n, n=n)

        n = 1000
        split = -1
        a = np.ones((n, 10, 2))
        b = np.ones((n, 20, 2)) * 2
        c = np.ones((n, 30, 2)) * 3
        d = np.ones((n, 30, 2)) * 3
        train_test_tuple = training.train_test_split(a, b, c, d, split=split)
        self.__assert(train_test_tuple, s=n, n=n)

        n = 100
        split = 0.5
        a = np.arange(n)
        train_test_tuple = training.train_test_split(a, split=split, shuffle=True, seed=0)
        self.__assert(train_test_tuple, s=split*n, n=n)

        b = np.hstack((train_test_tuple[0], train_test_tuple[1]))
        self.assertTrue(~np.allclose(a, b))
        b.sort()
        self.assertTrue(np.allclose(a, b))
