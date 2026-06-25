"""File containing support for typing."""

from collections.abc import Sequence
from typing import Callable, Protocol

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

Vec3Callable = Callable[[float], VecLike3]


class FlowConditionCallable(Protocol):
    """Protocol for flow condition functions."""

    def __call__(
        self,
        time: float,
        positions: npt.NDArray[np.double],
        out_array: npt.NDArray[np.double] | None,
    ) -> npt.ArrayLike:
        """Determine the flow condition at a given time.

        Parameters
        ----------
        time : float
            The current simulation time.

        positions : array
            Positions of the geometry points in the global coordinate system.

        out_array : array, optional
            Array to write the flow condition into. If None, should be created.

        Returns
        -------
        array_like
            The flow condition at the given time.
        """
        ...


FlowConditionSpecifications = FlowConditionCallable | float | npt.ArrayLike
