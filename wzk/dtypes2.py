import numpy as np


__str2np = {"f": np.float64,
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
            "cmp": object,  # byte strings can have varying length and otherwise there is a fuck-up
            # 'zlib': object,  # does this make sense to add?
            "t": str,
            "txt": str,
            "str": str,
            }


def str2np(s: str, strip: bool = True):
    if strip:
        s = s.split("_")
        if len(s) == 1:
            return None
        else:
            s = s[-1]
    return __str2np[s]


def astype(a, s):
    return a.astype(str2np(s))


c2np = {bool: np.bool_,
        str: np.str_,
        int: np.integer,
        float: np.floating}
