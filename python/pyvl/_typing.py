"""File containing support for typing."""

from collections.abc import Sequence
from typing import Callable

import numpy as np
from numpy import typing as npt

FloatLike = int | float
VecLike3 = (
    tuple[FloatLike, FloatLike, FloatLike]
    | Sequence[FloatLike]
    | npt.NDArray[np.floating]
    | npt.NDArray[np.integer]
)

CallableSerializer = Callable[[Callable], str]
CallableDeserializer = Callable[[str], Callable]
