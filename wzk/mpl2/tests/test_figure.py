from unittest import TestCase
import os

from wzk.mpl2.figure import new_fig, save_fig
from wzk import files

_dir = os.path.abspath(os.path.dirname(__file__)) + "/"
_file = "temp__test_mpl_latex"


class Test(TestCase):
    def test_save_fig(self):
        file = f"{_dir}{_file}"
        fig, ax = new_fig()
        save_fig(file=file, fig=fig, formats=("png",), show=False)
        files.rmdirs(file)
        self.assertTrue(True)
