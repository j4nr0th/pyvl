"""Implementation of the flow solver settings."""

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt

from pyvl._typing import (
    CallableDeserializer,
    CallableSerializer,
    FlowConditionSpecifications,
)
from pyvl.cvl import TransformationPlane
from pyvl.fio.io_common import HirearchicalMap
from pyvl.geometry import SimulationGeometry


class ShedderCallback(Protocol):
    """Protocol for wake shedding functions."""

    def __call__(
        self,
        geometry: SimulationGeometry,
        positions: npt.NDArray[np.double],
        velocities: npt.NDArray[np.double],
        time: float,
    ) -> npt.ArrayLike:
        """Determine what elements should shed vorticity from the geometry.

        Parameters
        ----------
        geometry : SimulationGeometry
            The geometry to shed the wake from.

        positions : array
            Positions of the geometry points in the global coordinate system.

        velocities : array
            Velocities of the geometry points in the global coordinate system.

        time : float
            The current simulation time.

        Returns
        -------
        array_like
            Indices of lines which should shed vorticity from the geometry.
        """
        ...


@dataclass(frozen=True)
class WakeShedderCallback:
    """Dataclass specifying wake shedding based on a callback."""

    shedder: ShedderCallback
    """Function to determine which elements should shed vorticity from the geometry."""


@dataclass(frozen=True)
class WakeShedderUniform:
    """Dataclass specifying uniform wake shedding."""

    indices: npt.NDArray[np.uint]

    def __init__(self, indices: npt.ArrayLike) -> None:
        idx = np.asarray(indices, dtype=np.uint)
        if idx.ndim != 1:
            raise ValueError("Indices must be a 1D array.")
        object.__setattr__(self, "indices", idx)


@dataclass(frozen=True)
class WakeSettings:
    """Dataclass for wake model settings."""

    wake_shedder: WakeShedderUniform | WakeShedderCallback | None = None
    """Function to determine which elements should shed vorticity from the geometry."""

    wake_element_capacity: int = 1000
    """The maximum number of wake elements to store in the wake model."""

    def save(self, serializer: CallableSerializer) -> HirearchicalMap:
        """Serialize the object into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`WakeSettings` object.
        """
        hm = HirearchicalMap()
        hm.insert_int("wake_capacity", self.wake_element_capacity)
        if self.wake_shedder is not None:
            if isinstance(self.wake_shedder, WakeShedderUniform):
                hm.insert_string("type", "uniform")
                hm.insert_array("indices", self.wake_shedder.indices)
            else:
                hm.insert_string("type", "callback")
                hm.insert_string("callable", serializer(self.wake_shedder.shedder))
        return hm

    @classmethod
    def load(cls, hmap: HirearchicalMap, deserializer: CallableDeserializer) -> Self:
        """Deserialize the object from a HirearchicalMap."""
        if "type" in hmap:
            shedder_type = hmap.get_string("type")
            match shedder_type:
                case "uniform":
                    indices = hmap.get_array("indices")
                    wake_shedder = WakeShedderUniform(indices)
                case "callback":
                    callable_name = hmap.get_string("callable")
                    shedder_callable = deserializer(callable_name)
                    wake_shedder = WakeShedderCallback(shedder_callable)
                case _:
                    raise ValueError(f"Unknown shedder type: {shedder_type}")
        else:
            wake_shedder = None
        wake_capacity = hmap.get_int("wake_capacity")
        return cls(wake_shedder=wake_shedder, wake_element_capacity=wake_capacity)


@dataclass
class ModelSettings:
    """Class for specifying model settings.

    Parameters
    ----------
    vortex_cutoff : float
        Minimum normal distance before clamping velocity to zero.

    vortex_far_approximation : float
        Limit for applying arctan far field approximation.

    vortex_smallest_size : float
        Minimum line length below which execution is skipped.

    wake_element_capacity : int, default: 1000
        The maximum number of wake elements to store in the wake model.
    """

    vortex_cutoff: float
    """This sets the minimum normal distance at which any velocity is still induced."""

    vortex_far_approximation: float
    """This sets the limit/threshold for using the arctangent approximation."""

    vortex_smallest_size: float
    """This sets the minimum line length/filament size."""

    symmetry_plane: TransformationPlane | None = None
    """Optional symmetry plane used when computing induced velocities."""

    def save(self) -> HirearchicalMap:
        """Serialize the object into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`ModelSettings` object.
        """
        hm = HirearchicalMap()
        hm.insert_scalar("vortex_cutoff", self.vortex_cutoff)
        hm.insert_scalar("vortex_far_approximation", self.vortex_far_approximation)
        hm.insert_scalar("vortex_smallest_size", self.vortex_smallest_size)
        return hm

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize the object from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            Serialized state of the :class:`ModelSettings` object created by a call
            to :meth:`ModelSettings.save`.

        Returns
        -------
        Self
            Deserialized :class:`ModelSettings` object.
        """
        return cls(
            vortex_cutoff=hmap.get_scalar("vortex_cutoff"),
            vortex_far_approximation=hmap.get_scalar("vortex_far_approximation"),
            vortex_smallest_size=hmap.get_scalar("vortex_smallest_size"),
        )


@dataclass(frozen=True)
class SolverSettings:
    """Dataclass for solver settings.

    Parameters
    ----------
    flow_conditions : FlowConditions
        Flow conditions to use for the solver.

    model_settings : ModelSettings
        Settings for the models used by the solver.

    wake_settings : WakeSettings, optional
        Settings for the wake model used by the solver.
        If not specified, no wake will be shed.
    """

    model_settings: ModelSettings
    flow_velocity: FlowConditionSpecifications | None = None
    wake_settings: WakeSettings = WakeSettings()

    def __post_init__(self) -> None:
        """Check that the flow conditions are valid."""
        if self.flow_velocity is not None:
            if not callable(self.flow_velocity):
                v = np.array(self.flow_velocity, dtype=np.float64).reshape(3)
                object.__setattr__(self, "flow_velocity", v)

    def save(self, serializer: CallableSerializer) -> HirearchicalMap:
        """Serialize the object into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`SolverSettings` object.
        """
        hm = HirearchicalMap()
        # Flow conditiotns
        if self.flow_velocity is not None:
            if callable(self.flow_velocity):
                hm.insert_string("flow_velocity_callable", serializer(self.flow_velocity))
            else:
                hm.insert_array("flow_velocity_constant", self.flow_velocity)

        # Model settings
        hm.insert_hirearchical_map("model_settings", self.model_settings.save())
        # Wake settings
        hm.insert_hirearchical_map("wake_settings", self.wake_settings.save(serializer))
        return hm

    @classmethod
    def load(cls, hmap: HirearchicalMap, deserializer: CallableDeserializer) -> Self:
        """Deserialize the object from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            Serialized state of the :class:`SolverSettings` object created by a call
            to :meth:`SolverSettings.save`.

        Returns
        -------
        Self
            Deserialized :class:`SolverSettings` object.
        """
        # Model settings
        model_settings = ModelSettings.load(hmap.get_hirearchical_map("model_settings"))
        # Wake settings
        wake_settings = WakeSettings.load(
            hmap.get_hirearchical_map("wake_settings"), deserializer
        )
        flow_vel: None | FlowConditionSpecifications = None
        if "flow_velocity_callable" in hmap:
            flow_vel = deserializer(hmap.get_string("flow_velocity_callable"))

        if "flow_velocity_constant" in hmap:
            assert flow_vel is None, (
                "Both flow_velocity_callable and flow_velocity_constant are specified."
            )
            flow_vel = hmap.get_array("flow_velocity_constant")

        return cls(
            flow_velocity=flow_vel,
            model_settings=model_settings,
            wake_settings=wake_settings,
        )

    def get_flow_velocity(
        self,
        time: float,
        positions: npt.ArrayLike,
        out_array: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        """Get the flow velocity at a given time and position.

        Parameters
        ----------
        time : float
            The current simulation time.

        positions : array_like
            Positions in the global coordinate system.

        out_array : array, optional
            Array to write the flow velocity into. If None, should be created.

        Returns
        -------
        array_like
            The flow velocity at the given time and position.
        """
        pos = np.asarray(positions, dtype=np.double)
        if pos.ndim <= 1 or pos.shape[-1] != 3:
            raise ValueError(
                "Position must be a 1D array of length 3 or a 2D array with shape (N, 3)."
            )
        if out_array is None:
            if self.flow_velocity is None:
                return np.zeros_like(pos, dtype=np.double)
            out_array = np.empty_like(pos, dtype=np.double)

        if self.flow_velocity is None:
            out_array.fill(0.0)
            return out_array

        if callable(self.flow_velocity):
            # Callable, so call it, then ensure the output has the correct type and shape
            return np.asarray(
                self.flow_velocity(time, pos, out_array), dtype=np.double
            ).reshape(pos.shape)
        else:
            vx, vy, vz = self.flow_velocity  # type: ignore __post_init__ ensures this is a 3-element array
            out_array[..., 0] = vx
            out_array[..., 1] = vy
            out_array[..., 2] = vz
            return out_array
