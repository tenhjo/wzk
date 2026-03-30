from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, Num

float32 = jnp.float32
int32 = jnp.int32

type Scalar = float | int | np.number[Any]
type ArrayLike = Array | np.ndarray[Any, Any]

type FloatArray = Float[Array, "..."]
type IntArray = Int[Array, "..."]
type BoolArray = Bool[Array, "..."]
type NumArray = Num[Array, "..."]

type ShapeLike = int | tuple[int, ...] | list[int]
type AxisLike = int | tuple[int, ...] | list[int] | None
