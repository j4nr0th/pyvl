"""Implementation of the flow solver settings."""

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from pyvl.fio.io_common import HirearchicalMap
from pyvl.flow_conditions import FlowConditions


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
    def simulation_times(self) -> npt.NDArray[np.float64]:
        """Times where simulation will run."""
        return np.arange(self.nt, dtype=np.float64) * np.float64(self.dt)

    @property
    def output_times(self) -> npt.NDArray[np.float64]:
        """Times where simulation will create output."""
        if self.output_interval is None or self.output_interval == 0:
            return self.simulation_times
        return np.float64(self.dt) * np.arange(
            self.nt, step=self.output_interval, dtype=np.float64
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
        return cls(nt=nt, dt=dt, output_interval=output_interval)


@dataclass
class ModelSettings:
    """Class for specifying model settings.

    Parameters
    ----------
    vortex_limit : float
        Minimum distance at which the vortex line induces and velocity.
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

    def save(self) -> HirearchicalMap:
        """Serialize the object into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`ModelSettings` object.
        """
        hm = HirearchicalMap()
        hm.insert_scalar("vortex_limit", self.vortex_limit)
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
        return cls(vortex_limit=hmap.get_scalar("vortex_limit"))


# TODO: symmetry settings


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

    wake_model : WakeModel, optional
        Model used to correct for the shedding of a wake from the geometry as
        a result of circulation. If not provided, no model will be used.
    """

    flow_conditions: FlowConditions
    model_settings: ModelSettings
    time_settings: TimeSettings = TimeSettings(1, 1, None)

    def save(self) -> HirearchicalMap:
        """Serialize the object into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`SolverSettings` object.
        """
        hm = HirearchicalMap()
        # Flow conditiotns
        fc = HirearchicalMap()
        fc.insert_type("type", type(self.flow_conditions))
        fc.insert_hirearchycal_map("data", self.flow_conditions.save())
        hm.insert_hirearchycal_map("flow_conditions", fc)
        # Model settings
        hm.insert_hirearchycal_map("model_settings", self.model_settings.save())
        # Time settings
        hm.insert_hirearchycal_map("time_settings", self.time_settings.save())
        return hm

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
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
        # Flow conditiotns
        fc = hmap.get_hirearchical_map("flow_conditions")
        flow_conditions_type: type[FlowConditions] = fc.get_type("type")
        flow_conditions: FlowConditions = flow_conditions_type.load(
            fc.get_hirearchical_map("data")
        )

        # Model settings
        model_settings = ModelSettings.load(hmap.get_hirearchical_map("model_settings"))
        # Time settings
        time_settings = TimeSettings.load(hmap.get_hirearchical_map("time_settings"))
        return cls(
            flow_conditions=flow_conditions,
            model_settings=model_settings,
            time_settings=time_settings,
        )
