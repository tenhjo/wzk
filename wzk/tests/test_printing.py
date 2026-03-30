
from wzk.logger import log_print
from unittest import TestCase

import numpy as np

from wzk import printing


class Test(TestCase):
    def test_print_stats(self):
        a = np.random.random(100) * 1
        b = np.random.random(100) * 2
        c = np.random.random(100) * 3
        printing.print_stats(a, b, c, names=("A", "BB", "CCC"))

        # self.fail()

    def test_print_table(self):
        printing.print_table(rows=["A", "BB", "CCC"], columns=["Some", "Random", "Words", "Foo"],
                             data=np.arange(12).reshape(3, 4), min_cell_size=10, cell_format=".4f")

    def test_print2(self):
        printing.print2("aaa", 1, 2, log_level=(1, 0))
        printing.print2(dict(b=1, bb=2), 11, 22, log_level=(1, 1))
        printing.print2("ccc", [3, "cc", 333], 33, log_level=(1, 2), sep="---")
        printing.print2("nice", "a", "staircase", log_level=(1, 1), sep="    ")
        printing.print2("back", "to", "level", "zero", log_level=(1, 0), sep="::")

    def test_print_array_3d(self):
        array_3d = np.arange(4 * 5 * 6).reshape((4, 5, 6))
        printing.print_array_3d(array_3d)

    def test_color(self):
        log_print("normal")
        log_print(printing.color_text(s="normal2", color="k", background="w"))
        log_print(printing.color_text(s="red", color="red", background="w"))
        log_print(printing.color_text(s="red", color="red", background="k"))
        log_print(printing.color_text(s="normal2", color="k", background="w"))
        log_print("normal")

        for c in ["w", "r", "g", "y", "b", "m", "c", "l", "k", "w"]:
            log_print(printing.color_text(s=c, color=c, background=c))

        for c in ["w", "r", "g", "y", "b", "m", "c", "l", "k", "w"]:
            log_print(printing.color_text(s=c, color=c, background="k"))

        for c in ["w", "r", "g", "y", "b", "m", "c", "l", "k", "w", "k"]:
            log_print(printing.color_text(s=c, color=c, background="w"))

    def test_save_string_concatenate(self):
        s = "test - full"
        prefix = "hello"
        suffix = "bye"
        delimiter = " | "
        assert printing.pre_string_suf(s=s, prefix=prefix, suffix=suffix,
                                       delimiter=delimiter) == f"{prefix}{delimiter}{s}{delimiter}{suffix}"

        s = ""
        prefix = "hello"
        suffix = "bye"
        delimiter = " / "
        assert printing.pre_string_suf(s=s, prefix=prefix, suffix=suffix,
                                       delimiter=delimiter) == f"{prefix}{delimiter}{suffix}"

        s = "test - only prefix"
        prefix = "hello"
        suffix = ""
        delimiter = " : "
        assert printing.pre_string_suf(s=s, prefix=prefix, suffix=suffix,
                                       delimiter=delimiter) == f"{prefix}{delimiter}{s}"

        s = "test - only suffix"
        prefix = ""
        suffix = "bye"
        delimiter = " : "
        assert printing.pre_string_suf(s=s, prefix=prefix, suffix=suffix,
                                       delimiter=delimiter) == f"{s}{delimiter}{suffix}"
