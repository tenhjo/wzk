from __future__ import annotations

import importlib
from typing import Any

from .environ import __multiprocessing2  # must be imported before multiprocessing / numpy  # noqa: F401

try:  # optional dependency
    from pyOpt.pySLSQP.pySLSQP import SLSQP as _  # noqa: F401
except ImportError:
    pass

from .time2 import (tic as tic,
                    toc as toc,
                    tictoc as tictoc,
                    get_timestamp as get_timestamp)

from .printing import (progress_bar as progress_bar,
                       print2 as print2,
                       check_verbosity as check_verbosity)


_LAZY_MODULES = {
    "files": "wzk.io.files",
    "sql2": "wzk.io.sql2",
    "math2": "wzk.math.math2",
    "geometry": "wzk.math.geometry",
    "jax2": "wzk.jax2",
    "random2": "wzk.random.random2",
    "perlin": "wzk.random.perlin",
    "alg": "wzk.alg",
    "limits": "wzk.limits",
    "opt": "wzk.opt",
    "strings": "wzk.strings",
    "image": "wzk.image",
    "bimage": "wzk.bimage",
    "grid": "wzk.grid",
    "ltd": "wzk.ltd",
}


__all__ = [
    "tic",
    "toc",
    "tictoc",
    "get_timestamp",
    "progress_bar",
    "print2",
    "check_verbosity",
    *_LAZY_MODULES.keys(),
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'wzk' has no attribute '{name}'")

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if name == "jax2":
            raise AttributeError(
                "module 'wzk' has no attribute 'jax2' (optional JAX dependencies are missing)"
            ) from exc
        raise

    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
