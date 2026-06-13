"""Test the solver module functions."""

from unittest.mock import patch

import numpy as np
import pytest
from pyvl.cvl import Mesh, ReferenceFrame, quad_induction
from pyvl.flow_conditions import FlowConditionsUniform
from pyvl.geometry import Geometry, SimulationGeometry
from pyvl.settings import (
    ModelSettings,
    SolverSettings,
    TimeSettings,
    WakeSettings,
    WakeShedderUniform,
)
from pyvl.solver import (
    OutputSettings,
    SolverResults,
    SolverState,
    _compute_induced_velocity,
    run_solver,
    update_simulation_state,
)
from pyvl.wake import WakeState


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
    model_settings = ModelSettings(vortex_limit=1e-6, wake_settings=wake_settings)
    time_settings = TimeSettings(nt=2, dt=0.1)
    settings = SolverSettings(flow_cond, model_settings, time_settings)

    return sim_geo, settings


def test_run_solver(basic_setup):
    """Check that the solver runs and produces results with the expected structure."""
    sim_geo, settings = basic_setup

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


def test_line_circulation():
    """Check we compute line circulations correctly using a mesh dual."""
    rng = np.random.default_rng(241)
    # The mesh represents this geometry:
    #
    # 0-----1-----3
    #  \ 0 / \ 1 /
    #   \ / 2 \ /
    #    2-----4
    #     \ 3 /
    #      \ /
    #       5
    #
    #
    msh = Mesh(
        n_points=6,
        connectivity=(
            (2, 1, 0),  # S0
            (3, 1, 4),  # S1
            (2, 4, 1),  # S2
            (4, 2, 5),  # S3
        ),
    )
    # Mesh lines are then:
    # Line | Start | End
    # =====+=======+====
    #  0   |   0   | 1
    #  1   |   1   | 2
    #  2   |   2   | 0
    #  3   |   3   | 1
    #  4   |   1   | 4
    #  5   |   4   | 3
    #  6   |   2   | 4
    #  7   |   5   | 2
    #  8   |   4   | 5
    circulations = rng.random(msh.n_surfaces)
    dual = msh.compute_dual()
    line_circ = dual.line_circulations(circulations)
    # This is manually computed based on how the lines are
    expected = np.array(
        (
            circulations[0],  # Line 0
            circulations[0] - circulations[2],  # Line 1
            circulations[0],  # Line 2
            circulations[1],  # Line 3
            circulations[1],  # Line 4
            circulations[1] - circulations[2],  # Line 5
            circulations[2] - circulations[3],  # Line 6
            circulations[3],  # Line 7
            circulations[3],  # Line 8
        )
    )
    assert pytest.approx(line_circ) == expected


def test_induction_two_triangles_equal_to_quad():
    """Check that we do induction velocity correctly."""
    rng = np.random.default_rng(203)
    # make two meshes:
    # - a big quad
    # - triangles which the first one is equivalent to
    qmsh = Mesh(
        4,
        ((0, 1, 2, 3),),
    )
    tmsh = Mesh(
        4,
        ((0, 1, 3), (1, 2, 3)),
    )
    positions = np.array(
        (
            (-1, -1, 0),
            (+1, -1, 0),
            (+1, +1, 0),
            (-1, +1, 0),
        ),
        np.double,
    )

    # make empty conditions
    empty_wake = WakeState.empty(1)
    some_flow_conditions = FlowConditionsUniform(vx=2, vy=3, vz=4, rho=1e-3, p_stat=5)
    # Check with constant circulation for both
    tol = 1e-6
    # Use 4 random induction values for circulation
    cv = rng.random(4)

    # Random numbers between -2 and +2 for positions
    tgt = rng.random((4, 2, 2, 3)) * 4 - 2

    velocity_q = _compute_induced_velocity(
        time=0,
        tol=tol,
        mesh=qmsh,
        positions=positions,
        line_circulation=cv,
        wake=empty_wake,
        flow_cond=some_flow_conditions,
        target=tgt,
    )
    assert velocity_q.shape == tgt.shape

    velocity_t = _compute_induced_velocity(
        time=0,
        tol=tol,
        mesh=tmsh,
        positions=positions,
        # The center line would be zero
        line_circulation=np.array((cv[0], cv[1], 0, cv[2], cv[3])),
        wake=empty_wake,
        flow_cond=some_flow_conditions,
        target=tgt,
    )
    assert velocity_t.shape == tgt.shape

    assert pytest.approx(velocity_q) == velocity_t


def test_wake_induction_same_as_mesh():
    """Verify that induction computed for wake is same as for equivalent mesh."""
    rng = np.random.default_rng(51)

    # Connectivity of the quads
    quad_conn = np.array(
        (
            (0, 1, 4, 3),
            (3, 4, 5, 6),
            (1, 2, 5, 4),
            (7, 8, 10, 9),
        ),
        np.uint32,
    )

    # Mesh of the quads
    msh = Mesh(n_points=11, connectivity=quad_conn)
    positions = rng.random((msh.n_points, 3))

    quad_pos = positions[quad_conn.reshape(-1), :].reshape(*quad_conn.shape, 3)
    circ = rng.random(msh.n_surfaces)

    wake = WakeState.empty(4)
    wake = wake.add_quads(quad_pos, circ)

    line_circ = msh.compute_dual().line_circulations(circ)

    # Compute induced velocity with both the mesh and the wake
    tol = 1e-6
    tgt = rng.random((33, 3)) * 4 - 2  # Random numbers between -2 and +2
    wake_ind = wake.induced_velocity(tol=tol, positions=tgt)

    mesh_ind = msh.induction_velocity(
        tol=tol, positions=positions, control_points=tgt, line_circulation=line_circ
    )

    assert pytest.approx(wake_ind) == mesh_ind


def test_state_update():
    """Ensure the state update computes the correct circulation."""
    rng = np.random.default_rng(241)
    # Positions of the quad corners
    pos = np.array(
        (
            (-1, -1, 0),
            (+1, -1, 0),
            (+1, +1, 0),
            (-1, +1, 0),
        )
    )
    # Normal to the quad
    normal = np.array((0, 0, 1))

    # Make sure the normal is actually normal to all edges
    for i in range(pos.shape[0]):
        p1 = pos[i]
        p2 = pos[(i + 1) % pos.shape[0]]
        d = p2 - p1
        assert np.isclose(np.dot(d, normal), 0)

    # Make the simulation geometry based on the QUAD
    sim_geo = SimulationGeometry.from_geometries(
        Geometry(
            label="the QUAD",
            reference_frame=ReferenceFrame(),
            mesh=Mesh(n_points=4, connectivity=((0, 1, 2, 3),)),
            positions=pos,
        )
    )
    # Pick some (pseudo) random flow conditions
    flow_conditions = FlowConditionsUniform(
        vx=rng.random(), vy=rng.random(), vz=rng.random()
    )
    # Settings have nothing interesting besides the flow conditions
    settings = SolverSettings(
        flow_conditions=flow_conditions, model_settings=ModelSettings(vortex_limit=1e-6)
    )
    # Create new empty state
    state = SolverState.create_new(time=0, geometry=sim_geo, settings=settings)

    # Compute the resulting state
    target_time = 1 + rng.random()  # should not matter, as long as more than start time
    # Compute the induction based on the no-penetration
    state = update_simulation_state(state, target_time=target_time, out_state=state)

    # Get induction directly from the QUAD
    tgt = np.mean(pos, axis=0)
    ind_vel = quad_induction(
        tol=settings.model_settings.vortex_limit,
        quad_positions=pos.reshape(1, 4, 3),
        quad_circulations=state.circulation,
        target_positions=tgt.reshape(1, 3),
    )
    # Get the flow velocity at the CP
    flow_vel = flow_conditions.get_velocity(time=target_time, positions=tgt)

    # Normal flow through CP should be (basically) zero for the no penetration condition
    assert np.isclose(np.dot(ind_vel + flow_vel, normal), 0)


if __name__ == "__main__":
    test_line_circulation()
    test_induction_two_triangles_equal_to_quad()
    test_wake_induction_same_as_mesh()
    test_state_update()
