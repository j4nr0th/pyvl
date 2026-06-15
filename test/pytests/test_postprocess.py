"""Test the postprocess module functions."""

from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from pyvl.cvl import Mesh, ReferenceFrame
from pyvl.flow_conditions import FlowConditionsUniform
from pyvl.geometry import Geometry, SimulationGeometry
from pyvl.postprocess import forces, pressure, velocity
from pyvl.settings import (
    ModelSettings,
    SolverSettings,
    TimeSettings,
    WakeSettings,
    WakeShedderUniform,
)
from pyvl.solver import SolverResults
from pyvl.wake import WakeState


@pytest.fixture
def mock_solver_results():
    """Set up a minimal geometry."""
    # A simple mesh: 4 points, 1 surface (2 triangles)
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.double)
    # Connectivity: 2 triangles (0,1,2) and (0,2,3)
    connectivity = [
        np.array([0, 1, 2], dtype=np.uint32),
        np.array([0, 2, 3], dtype=np.uint32),
    ]
    mesh = Mesh(len(points), connectivity)
    rf = ReferenceFrame()
    geo = Geometry("test_geo", rf, mesh, points)
    sim_geo = SimulationGeometry.from_geometries(geo)

    # Setup minimal settings
    flow_cond = FlowConditionsUniform(1.0, 0.0, 0.0)
    wake_settings = WakeSettings(WakeShedderUniform(np.array([0], dtype=np.uint)))
    model_settings = ModelSettings(vortex_limit=1e-3, wake_settings=wake_settings)
    time_settings = TimeSettings(nt=1, dt=1.0)
    settings = SolverSettings(flow_cond, model_settings, time_settings)

    wake_settings = WakeSettings(WakeShedderUniform(np.array([0], dtype=np.uint)))
    model_settings = ModelSettings(vortex_limit=1e-3, wake_settings=wake_settings)
    time_settings = TimeSettings(nt=1, dt=1.0)
    settings = SolverSettings(flow_cond, model_settings, time_settings)

    results = SolverResults(sim_geo, settings)
    results.circulations = np.zeros((1, sim_geo.n_surfaces), dtype=np.double)

    # Mock WakeState
    wake_state = MagicMock(spec=WakeState)

    def mock_induced_vel(
        tol: float,
        positions: npt.NDArray[np.double],
        out_velocity: npt.NDArray[np.double] | None = None,
        n_threads: int = 1,
    ) -> npt.NDArray[np.double]:
        """Mock method for induced velocity."""
        del tol, n_threads
        if out_velocity is None:
            out_velocity = np.empty_like(positions)

        out_velocity[:] = 0
        return out_velocity

    wake_state.induced_velocity.side_effect = mock_induced_vel

    results.wake_states = [wake_state]

    return results


def test_compute_surface_dynamic_pressure(mock_solver_results):
    """Check the surface pressure is computed and has the correct shape."""
    res = pressure.compute_surface_dynamic_pressure(mock_solver_results)
    assert isinstance(res, list)
    assert len(res) == 1
    assert isinstance(res[0], np.ndarray)


def test_compute_dynamic_pressure_variable(mock_solver_results):
    """Check the variable pressure is computed and has the correct shape."""
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.double)
    res = pressure.compute_dynamic_pressure_variable(mock_solver_results, [pts])
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0].shape == (2,)


def test_compute_dynamic_pressure_variable_error(mock_solver_results):
    """Check that an error is raised if the input points have the wrong shape."""
    pts = np.array([0, 0, 0], dtype=np.double)  # Wrong shape
    with pytest.raises(
        ValueError, match="Positions must be an array of 3 component position vectors."
    ):
        pressure.compute_dynamic_pressure_variable(mock_solver_results, [pts])


def test_compute_velocities(mock_solver_results):
    """Check the velocities are computed and have the correct shape."""
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.double)
    res = velocity.compute_velocities(mock_solver_results, pts)
    assert res.shape == (1, 2, 3)


def test_compute_velocities_error(mock_solver_results):
    """Check that an error is raised if the input points have the wrong shape."""
    pts = np.array([0, 0, 0], dtype=np.double)
    with pytest.raises(
        ValueError, match="Positions must be an array of 3 component position vectors."
    ):
        velocity.compute_velocities(mock_solver_results, pts)


def test_compute_velocities_variable(mock_solver_results):
    """Check the variable velocities are computed and have the correct shape."""
    pts = [np.array([[0, 0, 0]], dtype=np.double), np.array([[1, 1, 1]], dtype=np.double)]

    # Create a new SolverResults with nt=2
    settings = mock_solver_results.settings
    new_time_settings = TimeSettings(nt=2, dt=1.0)
    # Since SolverSettings is frozen, we create a new one
    import dataclasses

    new_settings = dataclasses.replace(settings, time_settings=new_time_settings)

    # We need a new SolverResults because it uses the settings in __init__
    # But we can just mock the results object since we only need certain attributes
    results = MagicMock(spec=SolverResults)
    results.settings = new_settings
    results.geometry = mock_solver_results.geometry
    results.circulations = np.zeros(
        (2, mock_solver_results.geometry.n_surfaces), dtype=np.double
    )

    wake_state = MagicMock(spec=WakeState)
    wake_state.induced_velocity.side_effect = lambda tol, pts: (
        tol * np.zeros((len(pts), 3), dtype=np.double)
    )
    results.wake_states = [wake_state, wake_state]

    res = velocity.compute_velocities_variable(results, pts)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0].shape == (1, 3)


def test_compute_velocities_variable_error(mock_solver_results):
    """Check that an error is raised if the input points have the wrong shape."""
    pts = [np.array([0, 0, 0], dtype=np.double)]
    with pytest.raises(
        ValueError, match="Positions must be an array of 3 component position vectors."
    ):
        velocity.compute_velocities_variable(mock_solver_results, pts)


def test_circulatory_forces(mock_solver_results):
    """Check the circulatory forces are computed and have the correct shape."""
    # Mesh.line_forces is a static method, we can mock it or let it run
    # Since we have a real mesh, we let it run.
    res = forces.circulatory_forces(mock_solver_results)
    assert isinstance(res, list)
    assert len(res) == 1
