"""Test the solver module functions."""

from unittest.mock import patch

import numpy as np
import pytest
from pyvl.cvl import Mesh, ReferenceFrame
from pyvl.flow_conditions import FlowConditionsUniform
from pyvl.geometry import Geometry, SimulationGeometry
from pyvl.settings import (
    ModelSettings,
    SolverSettings,
    TimeSettings,
    WakeSettings,
    WakeShedderUniform,
)
from pyvl.solver import OutputSettings, SolverResults, run_solver, update_simulation_state


@pytest.fixture
def basic_setup():
    """Set up a minimal geometry."""
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.double)
    connectivity = [
        np.array([0, 1, 2], dtype=np.uint32),
        np.array([0, 2, 3], dtype=np.uint32),
    ]
    mesh = Mesh(len(points), connectivity)
    rf = ReferenceFrame()
    geo = Geometry("test_geo", rf, mesh, points)
    sim_geo = SimulationGeometry.from_geometries(geo)

    flow_cond = FlowConditionsUniform(1.0, 0.0, 0.0)
    wake_settings = WakeSettings(WakeShedderUniform(np.array([0], dtype=np.uint)))
    model_settings = ModelSettings(vortex_limit=1.0, wake_settings=wake_settings)
    time_settings = TimeSettings(nt=2, dt=0.1)
    settings = SolverSettings(flow_cond, model_settings, time_settings)

    return sim_geo, settings


def test_run_solver(basic_setup):
    """Check that the solver runs and produces results with the expected structure."""
    sim_geo, settings = basic_setup

    with (
        patch("scipy.linalg.lu_factor") as mock_lu_f,
        patch("scipy.linalg.lu_solve") as mock_lu_s,
    ):
        mock_lu_f.return_value = (
            np.eye(sim_geo.n_surfaces),
            np.ones(sim_geo.n_surfaces, dtype=int),
        )
        mock_lu_s.return_value = np.zeros(sim_geo.n_surfaces)
        results = run_solver(sim_geo, settings, None)
        assert isinstance(results, SolverResults)
        assert results.circulations.shape == (2, sim_geo.n_surfaces)
        assert len(results.wake_states) == 2


def test_run_solver_with_output(basic_setup, tmp_path):
    """Check that the solver runs and produces output files with output settings."""
    sim_geo, settings = basic_setup

    def naming_callback(i, _):
        return str(tmp_path / f"out_{i}.json")

    output_settings = OutputSettings("JSON", naming_callback)
    with (
        patch("scipy.linalg.lu_factor") as mock_lu_f,
        patch("scipy.linalg.lu_solve") as mock_lu_s,
    ):
        mock_lu_f.return_value = (
            np.eye(sim_geo.n_surfaces),
            np.ones(sim_geo.n_surfaces, dtype=int),
        )
        mock_lu_s.return_value = np.zeros(sim_geo.n_surfaces)
        results = run_solver(sim_geo, settings, output_settings)

    assert isinstance(results, SolverResults)
    assert (tmp_path / "out_0.json").exists()


def test_update_simulation_state_basic(basic_setup):
    """Check that the simulation state is updated correctly for a single time step."""
    sim_geo, settings = basic_setup
    from pyvl.solver import SolverState

    state = SolverState.create_new(0.0, sim_geo, settings)

    with (
        patch("scipy.linalg.lu_factor") as mock_lu_f,
        patch("scipy.linalg.lu_solve") as mock_lu_s,
    ):
        mock_lu_f.return_value = (
            np.eye(sim_geo.n_surfaces),
            np.ones(sim_geo.n_surfaces, dtype=int),
        )
        mock_lu_s.return_value = np.zeros(sim_geo.n_surfaces)
        new_state = update_simulation_state(state, 0.1)

    assert new_state.time == 0.1
    assert new_state.circulation.shape == (sim_geo.n_surfaces,)
    assert (
        new_state.wake.capacity
        == settings.model_settings.wake_settings.wake_element_capacity
    )
