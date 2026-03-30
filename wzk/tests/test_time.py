
from wzk.logger import log_print
from unittest import TestCase

from wzk import tic, toc, tictoc
from wzk import time2


class Test(TestCase):
    def test_tictoc(self):

        tic()  # 1
        time2.sleep(0.1)
        tic()  # 2
        time2.sleep(0.2)
        toc()  # 2
        time2.sleep(0.2)
        tic()  # 3
        time2.sleep(0.2)
        toc()  # 3
        time2.sleep(0.1)
        t = toc()  # 1
        self.assertTrue(t > 0.8)
        self.assertTrue(t < 0.9)

    def test_get_timestamp(self):
        s = time2.get_timestamp(year=True, month=True, day=True, hour=True, minute=True, second=True, millisecond=True,
                                date_separator=".", time_separator=":", date_time_separator=" ")
        log_print(s)
        s = time2.get_timestamp(year=True, month=False, day=False, hour=False, minute=False, second=False,
                                millisecond=False,
                                date_separator=".", time_separator=":", date_time_separator=" ")
        log_print(s)

    def test_tictoc2(self):
        with tictoc(text="Sample Text A", log_level=1) as _:
            time2.sleep(0.1)
            with tictoc(text="Sample Text B1", log_level=1) as _:
                time2.sleep(0.2)
            with tictoc(text="Sample Text B2", log_level=1) as _:
                time2.sleep(0.2)
