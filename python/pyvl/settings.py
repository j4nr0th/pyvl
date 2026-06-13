"""Implementation of the flow solver settings."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt

from pyvl._typing import CallableDeserializer, CallableSerializer
from pyvl.fio.io_common import HirearchicalMap
from pyvl.fio.type_resolution import flow_conditions_from_serial
from pyvl.flow_conditions import FlowConditions
from pyvl.geometry import SimulationGeometry


@dataclass(frozen=True)
class TimeSettings:
    """Dataclass containing time setting options.

    Parameters
    ----------
    nt : int
        Number of steps to run the simulations for.

    dt : float
        The increment for each time step.

    output_interval : int, optional
        If specified, simulation output will be save only after this many
        iterations have passed since the last output.
    """

    nt: int
    dt: float
    output_interval: int | None = None

    @property
    def simulation_times(self) -> npt.NDArray[np.double]:
        """Times where simulation will run."""
        return np.arange(self.nt, dtype=np.double) * np.double(self.dt)

    @property
    def output_times(self) -> npt.NDArray[np.double]:
        """Times where simulation will create output."""
        if self.output_interval is None or self.output_interval == 0:
            return self.simulation_times
        return np.double(self.dt) * np.arange(
            self.nt, step=self.output_interval, dtype=np.double
        )  # type: ignore

    def save(self) -> HirearchicalMap:
        """Serialize the object into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`TimeSettings` object.
        """
        hm = HirearchicalMap()
        hm.insert_int("nt", self.nt)
        hm.insert_scalar("dt", self.dt)
        if self.output_interval is not None:
            hm.insert_int("output_interval", self.output_interval)
        return hm

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize the object from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            Serialized state of the :class:`TimeSettings` object created by a call
            to :meth:`TimeSettings.save`.

        Returns
        -------
        Self
            Deserialized :class:`TimeSettings` object.
        """
        nt = hmap.get_int("nt")
        dt = hmap.get_scalar("dt")
        if "output_interval" in hmap:
            output_interval = hmap.get_int("output_interval")
        else:
            output_interval = None
        return cls(nt=nt, dt=dt, output_interval=output_interval)


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
    vortex_limit : float
        Minimum distance at which the vortex line induces and velocity.

    wake_element_capacity : int, default: 1000
        The maximum number of wake elements to store in the wake model.
    """

    vortex_limit: float
    """This sets the minimum distance at which any velocity is still induced.

    Examples
    --------
    The effect of this value can be best shown using the following snippet:

    .. jupyter-execute::

        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>>
        >>> vortex_limit = 2e-1
        >>> x = np.linspace(0.1, 2, 1001)
        >>> y = 1 / x
        >>>
        >>> plt.plot(x, y, label="no limit", linestyle="dashed")
        >>> plt.plot(x, y * (x >= vortex_limit), label="limit = $0.2$")
        >>> plt.legend()
        >>> plt.grid()
        >>> plt.xlim(0.1, 1)
        >>> plt.ylim(0, 10)
        >>> plt.show()

    This becomes important if the two panels of either geometry or wake approach each
    other, as the induction might become too large and make the results unstable.
    """

    wake_settings: WakeSettings = WakeSettings()

    def save(self, serializer: CallableSerializer) -> HirearchicalMap:
        """Serialize the object into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`ModelSettings` object.
        """
        hm = HirearchicalMap()
        hm.insert_scalar("vortex_limit", self.vortex_limit)
        hm.insert_hirearchical_map("wake_settings", self.wake_settings.save(serializer))
        return hm

    @classmethod
    def load(cls, hmap: HirearchicalMap, deserializer: CallableDeserializer) -> Self:
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
            vortex_limit=hmap.get_scalar("vortex_limit"),
            wake_settings=WakeSettings.load(
                hmap.get_hirearchical_map("wake_settings"), deserializer
            ),
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

    time_setting : TimeSettings, default : TimeSettings(1, 1, None)
        Time iterations at which to run the solver. By default, a single iteration
        at time :math:`t = 0` will be run and the result recorded.
    """

    flow_conditions: FlowConditions
    model_settings: ModelSettings
    time_settings: TimeSettings = TimeSettings(1, 1, None)

    def save(self, serializer: CallableSerializer) -> HirearchicalMap:
        """Serialize the object into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`SolverSettings` object.
        """
        hm = HirearchicalMap()
        # Flow conditiotns
        fc = HirearchicalMap()
        fc.insert_string(
            "type",
            type(self.flow_conditions).__module__
            + "."
            + type(self.flow_conditions).__name__,
        )
        fc.insert_hirearchical_map("data", self.flow_conditions.save())
        hm.insert_hirearchical_map("flow_conditions", fc)
        # Model settings
        hm.insert_hirearchical_map("model_settings", self.model_settings.save(serializer))
        # Time settings
        hm.insert_hirearchical_map("time_settings", self.time_settings.save())
        return hm

    @classmethod
    def load(
        cls,
        hmap: HirearchicalMap,
        deserializer: CallableDeserializer,
        custom_types: Mapping[str, type] | None = None,
        allow_override: bool = False,
    ) -> Self:
        """Deserialize the object from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            Serialized state of the :class:`SolverSettings` object created by a call
            to :meth:`SolverSettings.save`.
        custom_types : Mapping[str, type], optional
            A mapping of type names to types for custom subclasses of FlowConditions.
        allow_override : bool, default: False
            If True, custom types can override built-in types.

        Returns
        -------
        Self
            Deserialized :class:`SolverSettings` object.
        """
        fc = hmap.get_hirearchical_map("flow_conditions")
        flow_conditions = flow_conditions_from_serial(fc, custom_types, allow_override)

        # Model settings
        model_settings = ModelSettings.load(
            hmap.get_hirearchical_map("model_settings"), deserializer
        )
        # Time settings
        time_settings = TimeSettings.load(hmap.get_hirearchical_map("time_settings"))
        return cls(
            flow_conditions=flow_conditions,
            model_settings=model_settings,
            time_settings=time_settings,
        )
