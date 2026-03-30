from __future__ import annotations

import numpy as np


__str2np = {
    "f": np.float64,
    "f64": np.float64,
    "f32": np.float32,
    "f16": np.float16,
    "i": np.int32,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "ui64": np.uint64,
    "ui32": np.uint32,
    "ui16": np.uint16,
    "ui8": np.uint8,
    "b": bool,
    "cmp": object,
    "t": str,
    "txt": str,
    "str": str,
}


def str2np(s: str, strip: bool = True):
    if strip:
        parts = s.split("_")
        if len(parts) == 1:
            return None
        s = parts[-1]
    return __str2np[s]


def astype(a: np.ndarray, s: str) -> np.ndarray:
    return a.astype(str2np(s))


c2np = {
    bool: np.bool_,
    str: np.str_,
    int: np.integer,
    float: np.floating,
}
