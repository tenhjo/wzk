from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike as NpArrayLike

type ArrayLike = NpArrayLike
type ShapeLike = int | tuple[int, ...] | list[int]
type AxisLike = int | tuple[int, ...] | list[int] | None
type DTypeLike = np.dtype | type | None
