"""Implementation of the flow solver settings."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pydust.flow_conditions import FlowConditions
from pydust.wake import WakeModel


@dataclass(frozen=True)
class TimeSettings:
    """Dataclass containing time setting options."""

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


@dataclass
class ModelSettings:
    """Class for specifying model settings."""

    vortex_limit: float


# TODO: symmetry settings


@dataclass(frozen=True)
class SolverSettings:
    """Dataclass for solver settings."""

    flow_conditions: FlowConditions
    model_settings: ModelSettings
    time_settings: TimeSettings = TimeSettings(1, 1, None)
    wake_model: WakeModel | None = None
