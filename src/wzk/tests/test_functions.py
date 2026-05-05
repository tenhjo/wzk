import unittest

from wzk.functions import common_argument_wrapper


class Test(unittest.TestCase):
    def test_common_argument_wrapper(self):
        def fun(a, b, c, d, e, f, g):
            return a + b + c + d + e + f + g

        aa, bb, cc, dd, ee, ff, gg = 1, 2, 3, 4, 5, 6, 7

        fun_c = common_argument_wrapper(fun=fun, a=aa, b=bb, c=cc, d=dd)

        self.assertTrue(fun(a=aa, b=bb, c=cc, d=dd, e=ee, f=ff, g=gg) == fun_c(e=ee, f=ff, g=gg))
