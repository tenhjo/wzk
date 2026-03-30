
from wzk.logger import log_print
import os
import platform
import importlib.util
import matplotlib as mpl


headless = False


def __turn_on_headless():
    log_print("Matplotlib - Backend: 'headless' mode detected -> use 'Agg'")
    mpl.use("Agg")
    return True


def __has_qt_bindings() -> bool:
    return any(
        importlib.util.find_spec(module) is not None
        for module in ("PyQt6", "PySide6", "PyQt5", "PySide2")
    )


backend_override = os.environ.get("MPLBACKEND")

if backend_override:
    log_print(f"Matplotlib - Backend override from MPLBACKEND: '{backend_override}'")
    mpl.use(backend_override)

elif platform.system() == "Linux":
    try:
        display = os.environ["DISPLAY"]

        if "localhost" in display:
            headless = __turn_on_headless()
        else:
            mpl.use("TkAgg")

    except KeyError:
        headless = __turn_on_headless()

elif platform.system() == "Darwin":
    if __has_qt_bindings():
        mpl.use("QtAgg")  # Alternative for Mac: 'Qt5Agg', interplay with Pyvista often a bit tricky otherwise
        # mpl.use("macosx")
        # mpl.use("Qt5Agg")
    else:
        headless = __turn_on_headless()

import matplotlib.pyplot as plt  # noqa: F401, E402
