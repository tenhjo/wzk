import einops as eo
import numpy as np

from wzk import tictoc


def speed_einops():
    a = np.random.rand(10, 7, 3)

    with tictoc("a2") as _:
        pass

    with tictoc("a1") as _:
        eo.rearrange(a, "... -> ... ()")

    with tictoc("a2") as _:
        pass

    with tictoc("a1") as _:
        eo.rearrange(a, "... -> ... ()")
