from __future__ import annotations

from typing import Any, TypeAlias

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, Num

float32 = jnp.float32
int32 = jnp.int32

Scalar: TypeAlias = float | int | np.number[Any]
ArrayLike: TypeAlias = Any

FloatArray: TypeAlias = Float[Array, "..."]
IntArray: TypeAlias = Int[Array, "..."]
BoolArray: TypeAlias = Bool[Array, "..."]
NumArray: TypeAlias = Num[Array, "..."]

ShapeLike: TypeAlias = Any
AxisLike: TypeAlias = Any
