from __future__ import annotations

import importlib
from typing import Any

from .environ import __multiprocessing2  # must be imported before multiprocessing / numpy

try:  # optional dependency
    from pyOpt.pySLSQP.pySLSQP import SLSQP as _
except ImportError:
    pass

from .printing import check_verbosity as check_verbosity
from .printing import print2 as print2
from .printing import progress_bar as progress_bar
from .time2 import get_timestamp as get_timestamp
from .time2 import tic as tic
from .time2 import tictoc as tictoc
from .time2 import toc as toc

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
    "wandb2": "wzk.wandb2",
}


__all__ = [
    "tic",
    "toc",
    "tictoc",
    "get_timestamp",
    "new_fig",
    "progress_bar",
    "print2",
    "check_verbosity",
    *_LAZY_MODULES.keys(),
]


def new_fig(*args: Any, **kwargs: Any) -> Any:
    from wzk.mpl2.figure import new_fig as _new_fig
    return _new_fig(*args, **kwargs)


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
        if name == "wandb2":
            raise AttributeError(
                "module 'wzk' has no attribute 'wandb2' (install with: uv pip install 'wzk[wandb]')"
            ) from exc
        raise

    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
