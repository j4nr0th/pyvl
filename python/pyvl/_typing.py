"""File containing support for typing."""

from collections.abc import Sequence

import numpy as np
from numpy import typing as npt

FloatLike = int | float
VecLike3 = (
    tuple[FloatLike, FloatLike, FloatLike]
    | Sequence[FloatLike]
    | npt.NDArray[np.floating]
    | npt.NDArray[np.integer]
)
