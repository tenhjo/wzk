import os
import platform
import matplotlib as mpl


headless = False


def __turn_on_headless():
    print("Matplotlib - Backend: 'headless' mode detected -> use 'Agg'")
    mpl.use("Agg")
    return True


if platform.system() == "Linux":
    try:
        display = os.environ["DISPLAY"]

        if "localhost" in display:
            headless = __turn_on_headless()
        else:
            mpl.use("TkAgg")

    except KeyError:
        headless = __turn_on_headless()

elif platform.system() == "Darwin":
    # mpl.use('TkAgg')  # Alternative for Mac: 'Qt5Agg', interplay with Pyvista often a bit tricky otherwise
    mpl.use("macosx")
    # mpl.use("Qt5Agg")

import matplotlib.pyplot as plt  # noqa: F401, E402
