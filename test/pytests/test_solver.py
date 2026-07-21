"""Test the solver module functions."""

import numpy as np
import numpy.typing as npt
import pytest
from pyvl.cvl import (
    Mesh,
    ReferenceFrame,
    TransformationPlane,
    line_induction,
    line_normal_induction,
    quad_induction,
    quad_normal_induction,
)
from pyvl.geometry import Geometry, SimulationGeometry
from pyvl.settings import (
    ModelSettings,
    SolverSettings,
    WakeSettings,
    WakeShedderUniform,
)
from pyvl.solver import (
    OutputSettings,
    SolverState,
    SolverSystem,
    _compute_induced_velocity,
    _stitch_repeated_shed_quads,
    run_solver,
    update_simulation_state,
)
from pyvl.wake import WakeState


def _random_flow_velocity(
    time: float,
    positions: npt.NDArray[np.double],
    out_array: npt.NDArray[np.double] | None = None,
):
    """Return random flow velocity for testing."""
    del time
    if out_array is None:
        out_array = np.empty_like(positions)
    # Some random function
    out_array[:] = 3 * positions**2 - 2 * positions + 1
    return out_array


def make_square_geometry(
    label: str = "test_geo", reference_frame: ReferenceFrame | None = None
) -> Geometry:
    """Create a simple two-triangle square geometry."""
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.double)
    connectivity = [
        np.array([0, 1, 2], dtype=np.uint32),
        np.array([0, 2, 3], dtype=np.uint32),
    ]
    mesh = Mesh(len(points), connectivity)
    return Geometry(label, reference_frame or ReferenceFrame(), mesh, points)


@pytest.fixture
def basic_setup():
    """Set up a minimal geometry."""
    geo = make_square_geometry()
    sim_geo = SimulationGeometry.from_geometries(geo)

    wake_settings = WakeSettings(WakeShedderUniform(np.array([0], dtype=np.uint)))
    model_settings = ModelSettings(
        vortex_cutoff=1e-6,
        vortex_far_approximation=1e-6,
        vortex_smallest_size=1e-6,
    )
    settings = SolverSettings(
        model_settings=model_settings,
        flow_velocity=(1.0, 0.0, 0.0),
        wake_settings=wake_settings,
    )

    return sim_geo, settings


def test_run_solver(basic_setup):
    """Check that the solver runs and produces results with the expected structure."""
    sim_geo, settings = basic_setup

    results = run_solver(sim_geo, settings, times=[0])
    assert isinstance(results, tuple) and len(results) == 1
    assert isinstance(results[0], SolverState)


def test_solver_system_forwards_symmetry_plane():
    """Check the solver system constructor uses the symmetry plane."""
    plane = TransformationPlane((0.25, 0.0, 0.0), (1.0, 0.0, 0.0))
    geometry = make_square_geometry()

    system_with_plane = SolverSystem(
        time=0.0,
        vortex_cutoff=1e-6,
        vortex_far_approximation=1e-6,
        vortex_smallest_size=1e-6,
        geo=[geometry],
        symmetry_plane=plane,
    )
    system_without_plane = SolverSystem(
        time=0.0,
        vortex_cutoff=1e-6,
        vortex_far_approximation=1e-6,
        vortex_smallest_size=1e-6,
        geo=[geometry],
        symmetry_plane=None,
    )

    assert system_with_plane.symmetry_plane is plane
    assert not np.allclose(
        system_with_plane._self_induction_diags[geometry.label],
        system_without_plane._self_induction_diags[geometry.label],
    )


def test_solver_system_updates_self_diagonal_on_global_motion():
    """Check diagonal blocks are refreshed when a part moves globally under symmetry."""
    plane = TransformationPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    moving_frame = ReferenceFrame(theta=lambda t: (0.0, 0.0, t))
    geometry = make_square_geometry(reference_frame=moving_frame)

    v_co = 1e-6
    v_fa = 1e-6
    v_ss = 1e-6

    system_with_plane = SolverSystem(
        time=0.0,
        vortex_cutoff=v_co,
        vortex_far_approximation=v_fa,
        vortex_smallest_size=v_ss,
        geo=[geometry],
        symmetry_plane=plane,
    )
    before_with_plane = system_with_plane._self_induction_diags[geometry.label].copy()
    system_with_plane.update(
        1.0, vortex_cutoff=v_co, vortex_far_approximation=v_fa, vortex_smallest_size=v_ss
    )

    system_without_plane = SolverSystem(
        time=0.0,
        vortex_cutoff=v_co,
        vortex_far_approximation=v_fa,
        vortex_smallest_size=v_ss,
        geo=[geometry],
        symmetry_plane=None,
    )
    before_without_plane = system_without_plane._self_induction_diags[
        geometry.label
    ].copy()
    system_without_plane.update(
        1.0, vortex_cutoff=v_co, vortex_far_approximation=v_fa, vortex_smallest_size=v_ss
    )

    np.testing.assert_array_equal(
        before_without_plane, system_without_plane._self_induction_diags[geometry.label]
    )
    assert not np.array_equal(
        before_with_plane, system_with_plane._self_induction_diags[geometry.label]
    )


def test_compute_induced_velocity_forwards_symmetry_plane(basic_setup):
    """Check the induced-velocity helper forwards the symmetry plane to both calls."""
    sim_geo, settings = basic_setup
    plane = TransformationPlane((0.25, 0.0, 0.0), (1.0, 0.0, 0.0))
    settings.model_settings.symmetry_plane = plane

    geometry = make_square_geometry()
    positions = geometry.positions
    target = np.array([[0.25, 0.25, 0.25]], dtype=np.double)
    line_circulation = geometry.msh.line_circulations(np.array([1.0, -0.5]))
    wake = WakeState.empty(1).add_quads(
        new_positions=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=np.double,
        ),
        new_circulations=np.array([1.0], dtype=np.double),
    )

    result = _compute_induced_velocity(
        time=0.0,
        vortex_cutoff=settings.model_settings.vortex_cutoff,
        vortex_far_approximation=settings.model_settings.vortex_far_approximation,
        vortex_smallest_size=settings.model_settings.vortex_smallest_size,
        mesh=geometry.msh,
        positions=positions,
        line_circulation=line_circulation,
        wake=wake,
        flow_velocity=None,
        target=target,
        symmetry_plane=plane,
    )

    mesh_induced = geometry.msh.induction_velocity(
        vortex_cutoff=settings.model_settings.vortex_cutoff,
        vortex_far_approximation=settings.model_settings.vortex_far_approximation,
        vortex_smallest_size=settings.model_settings.vortex_smallest_size,
        positions=positions,
        control_points=target,
        line_circulation=line_circulation,
        symmetry_plane=plane,
    )
    wake_induced = wake.induced_velocity(
        vortex_cutoff=settings.model_settings.vortex_cutoff,
        vortex_far_approximation=settings.model_settings.vortex_far_approximation,
        vortex_smallest_size=settings.model_settings.vortex_smallest_size,
        positions=target,
        symmetry_plane=plane,
    )

    assert result.shape == target.shape
    np.testing.assert_allclose(result, mesh_induced + wake_induced)


def test_run_solver_with_output(basic_setup, tmp_path):
    """Check that the solver runs and produces output files with output settings."""
    sim_geo, settings = basic_setup

    def naming_callback(i, _):
        return str(tmp_path / f"out_{i}.json")

    output_settings = OutputSettings.new_python("JSON", naming_callback)
    results = run_solver(sim_geo, settings, times=[0], output_settings=output_settings)

    assert isinstance(results[0], SolverState)
    assert (tmp_path / "out_0.json").exists()


def test_update_simulation_state_basic(basic_setup):
    """Check that the simulation state is updated correctly for a single time step."""
    sim_geo, settings = basic_setup

    state = SolverState.create_new(0.0, sim_geo, settings)
    new_state = update_simulation_state(state, 0.1)

    assert new_state.time == 0.1
    assert new_state.circulation.shape == (sim_geo.n_lines,)
    assert new_state.wake.capacity == settings.wake_settings.wake_element_capacity


def test_update_simulation_state_forwards_symmetry_plane():
    """Check diagonal blocks refresh during an update when symmetry is active."""
    plane = TransformationPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    moving_frame = ReferenceFrame(theta=lambda t: (0.0, 0.0, t))
    geometry = make_square_geometry(reference_frame=moving_frame)
    sim_geo = SimulationGeometry.from_geometries(geometry)
    settings = SolverSettings(
        flow_velocity=(0.0, 0.0, 1.0),
        model_settings=ModelSettings(
            vortex_cutoff=1e-6,
            vortex_far_approximation=1e-6,
            vortex_smallest_size=1e-6,
            symmetry_plane=plane,
        ),
        wake_settings=WakeSettings(None, 1),
    )
    state = SolverState.create_new(0.0, sim_geo, settings)
    system = SolverSystem(
        time=0.0,
        vortex_cutoff=settings.model_settings.vortex_cutoff,
        vortex_far_approximation=settings.model_settings.vortex_far_approximation,
        vortex_smallest_size=settings.model_settings.vortex_smallest_size,
        geo=[geometry],
        symmetry_plane=plane,
    )

    before = system._self_induction_diags[geometry.label].copy()
    new_state = update_simulation_state(state, 1.0, system=system)

    assert new_state.time == 1.0
    assert not np.array_equal(before, system._self_induction_diags[geometry.label])


def test_stitch_repeated_shed_quads_uses_advected_previous_wake():
    """Check repeated shed lines stitch to the advected wake quad from last step."""
    wake = WakeState(
        quad_positions=np.array(
            [
                [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
                [[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [22.0, 0.0, 0.0], [23.0, 0.0, 0.0]],
                [[30.0, 0.0, 0.0], [31.0, 0.0, 0.0], [32.0, 0.0, 0.0], [33.0, 0.0, 0.0]],
            ],
            dtype=np.double,
        ),
        quad_circulations=np.array([1.0, 2.0, 3.0], dtype=np.double),
        quad_count=3,
        next_insertion_index=3,
    )
    previous_shed_lines = np.array([2, 5, 7], dtype=np.intp)
    updated_wake = wake.update_wake(
        dt=0.5,
        velocities=np.array(
            [
                [[1.0, 0.0, 0.0]] * 4,
                [[2.0, 0.0, 0.0]] * 4,
                [[3.0, 0.0, 0.0]] * 4,
            ],
            dtype=np.double,
        ),
    )
    new_quads = np.zeros((2, 4, 3), dtype=np.double)
    shedding_lines = np.array([5, 8], dtype=np.intp)

    _stitch_repeated_shed_quads(
        new_quads=new_quads,
        shedding_lines=shedding_lines,
        wake=wake,
        previous_shed_lines=previous_shed_lines,
        updated_wake=updated_wake,
    )

    np.testing.assert_array_equal(
        new_quads[0, 2:, :], updated_wake.quad_positions[1, 1::-1, :]
    )
    np.testing.assert_array_equal(new_quads[1], np.zeros((4, 3), dtype=np.double))


def test_stitch_repeated_shed_quads_wraps_ring_buffer():
    """Check repeated shed lines still stitch correctly after the wake wraps."""
    first_batch = np.array(
        [
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
            [[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [22.0, 0.0, 0.0], [23.0, 0.0, 0.0]],
        ],
        dtype=np.double,
    )
    second_batch = np.array(
        [
            [[30.0, 0.0, 0.0], [31.0, 0.0, 0.0], [32.0, 0.0, 0.0], [33.0, 0.0, 0.0]],
            [[40.0, 0.0, 0.0], [41.0, 0.0, 0.0], [42.0, 0.0, 0.0], [43.0, 0.0, 0.0]],
        ],
        dtype=np.double,
    )

    wake = WakeState.empty(3)
    wake = wake.add_quads(first_batch, np.array([1.0, 2.0], dtype=np.double))
    wake = wake.add_quads(second_batch, np.array([3.0, 4.0], dtype=np.double))
    assert wake.quad_count == 3
    assert wake.capacity == 3
    assert wake.next_insertion_index == 1

    updated_wake = wake.update_wake(
        dt=0.5,
        velocities=np.array(
            [
                [[1.0, 0.0, 0.0]] * 4,
                [[2.0, 0.0, 0.0]] * 4,
                [[3.0, 0.0, 0.0]] * 4,
            ],
            dtype=np.double,
        ),
    )

    new_quads = np.zeros((2, 4, 3), dtype=np.double)
    shedding_lines = np.array([4, 8], dtype=np.intp)

    _stitch_repeated_shed_quads(
        new_quads=new_quads,
        shedding_lines=shedding_lines,
        wake=wake,
        previous_shed_lines=np.array([4, 8], dtype=np.intp),
        updated_wake=updated_wake,
    )

    np.testing.assert_array_equal(
        new_quads[0, 2:, :], updated_wake.quad_positions[2, 1::-1, :]
    )
    np.testing.assert_array_equal(
        new_quads[1, 2:, :], updated_wake.quad_positions[0, 1::-1, :]
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
    line_circ = msh.line_circulations(circulations)
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
    # Check with constant circulation for both
    tol = 1e-6
    # Use 4 random induction values for circulation
    cv = rng.random(4)

    # Random numbers between -2 and +2 for positions
    tgt = rng.random((4, 2, 2, 3)) * 4 - 2

    velocity_q = _compute_induced_velocity(
        time=0,
        vortex_cutoff=tol,
        vortex_far_approximation=tol,
        vortex_smallest_size=tol,
        mesh=qmsh,
        positions=positions,
        line_circulation=cv,
        wake=empty_wake,
        flow_velocity=_random_flow_velocity,
        target=tgt,
    )
    assert velocity_q.shape == tgt.shape

    velocity_t = _compute_induced_velocity(
        time=0,
        vortex_cutoff=tol,
        vortex_far_approximation=tol,
        vortex_smallest_size=tol,
        mesh=tmsh,
        positions=positions,
        # The center line would be zero
        line_circulation=np.array((cv[0], cv[1], 0, cv[2], cv[3])),
        wake=empty_wake,
        flow_velocity=_random_flow_velocity,
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

    line_circ = msh.line_circulations(circ)

    # Compute induced velocity with both the mesh and the wake
    tol = 1e-6
    tgt = rng.random((33, 3)) * 4 - 2  # Random numbers between -2 and +2
    wake_ind = wake.induced_velocity(
        vortex_cutoff=tol,
        vortex_far_approximation=tol,
        vortex_smallest_size=tol,
        positions=tgt,
    )

    mesh_ind = msh.induction_velocity(
        vortex_cutoff=tol,
        vortex_far_approximation=tol,
        vortex_smallest_size=tol,
        positions=positions,
        control_points=tgt,
        line_circulation=line_circ,
    )

    assert pytest.approx(wake_ind) == mesh_ind


def test_quad_induction_symmetry_plane_matches_mirrored_copy() -> None:
    """Check quad induction with symmetry matches an explicit mirrored copy."""
    rng = np.random.default_rng(203)
    plane = TransformationPlane(origin=rng.random(3), normal=rng.random(3))
    quad_positions = rng.random((1, 4, 3)) * 4 - 2  # Random numbers between -2 and +2
    quad_circulation = rng.random(1)
    target_positions = rng.random((5, 3)) * 4 - 2  # Random numbers between -2 and +2

    base = quad_induction(
        vortex_cutoff=1e-8,
        vortex_far_approximation=1e-8,
        vortex_smallest_size=1e-8,
        quad_positions=quad_positions,
        quad_circulations=quad_circulation,
        target_positions=target_positions,
    )
    mirrored = quad_induction(
        vortex_cutoff=1e-8,
        vortex_far_approximation=1e-8,
        vortex_smallest_size=1e-8,
        quad_positions=plane.reflect(quad_positions),
        quad_circulations=-quad_circulation,
        target_positions=target_positions,
    )

    with_symmetry = quad_induction(
        vortex_cutoff=1e-8,
        vortex_far_approximation=1e-8,
        vortex_smallest_size=1e-8,
        quad_positions=quad_positions,
        quad_circulations=quad_circulation,
        target_positions=target_positions,
        symmetry_plane=plane,
    )

    expected = base + mirrored

    np.testing.assert_allclose(with_symmetry, expected, rtol=1e-12, atol=1e-12)


def test_quad_normal_induction_symmetry_plane_matches_mirrored_copy() -> None:
    """Check quad normal induction with symmetry matches an explicit mirrored copy."""
    rng = np.random.default_rng(204)
    plane = TransformationPlane(origin=rng.random(3), normal=rng.random(3))
    quad_positions = rng.random((1, 4, 3)) * 4 - 2  # Random numbers between -2 and +2
    quad_circulation = rng.random(1)
    N_TARGET = 10
    target_positions = (
        rng.random((N_TARGET, 3)) * 4 - 2
    )  # Random numbers between -2 and +2
    target_normals = rng.random((N_TARGET, 3)) * 2 - 1  # Random numbers between -1 and +1

    base = quad_normal_induction(
        vortex_cutoff=1e-8,
        vortex_far_approximation=1e-8,
        vortex_smallest_size=1e-8,
        quad_positions=quad_positions,
        quad_circulations=quad_circulation,
        target_positions=target_positions,
        target_normals=target_normals,
    )
    mirrored_target_only = quad_normal_induction(
        vortex_cutoff=1e-8,
        vortex_far_approximation=1e-8,
        vortex_smallest_size=1e-8,
        quad_positions=plane.reflect(quad_positions),
        quad_circulations=-quad_circulation,
        target_positions=target_positions,
        target_normals=target_normals,
    )

    with_symmetry = quad_normal_induction(
        vortex_cutoff=1e-8,
        vortex_far_approximation=1e-8,
        vortex_smallest_size=1e-8,
        quad_positions=quad_positions,
        quad_circulations=quad_circulation,
        target_positions=target_positions,
        target_normals=target_normals,
        symmetry_plane=plane,
    )

    expected = base + mirrored_target_only

    np.testing.assert_allclose(with_symmetry, expected, rtol=1e-12, atol=1e-12)


def test_quad_normal_induction_matches_velocity_projection() -> None:
    """Check quad normal induction matches projected quad induction."""
    rng = np.random.default_rng(207)
    quad_positions = rng.random((3, 4, 3)) * 4 - 2
    quad_circulation = rng.random(3) * 2 - 1
    target_positions = rng.random((9, 3)) * 4 - 2
    target_normals = rng.random((9, 3)) * 2 - 1
    target_normals /= np.linalg.norm(target_normals, axis=1, keepdims=True)
    plane = TransformationPlane(origin=rng.random(3), normal=rng.random(3))

    for symmetry_plane in (None, plane):
        velocity = quad_induction(
            vortex_cutoff=1e-8,
            vortex_far_approximation=1e-8,
            vortex_smallest_size=1e-8,
            quad_positions=quad_positions,
            quad_circulations=quad_circulation,
            target_positions=target_positions,
            symmetry_plane=symmetry_plane,
        )
        normal_velocity = quad_normal_induction(
            vortex_cutoff=1e-8,
            vortex_far_approximation=1e-8,
            vortex_smallest_size=1e-8,
            quad_positions=quad_positions,
            quad_circulations=quad_circulation,
            target_positions=target_positions,
            target_normals=target_normals,
            symmetry_plane=symmetry_plane,
        )

        expected = np.sum(velocity * target_normals, axis=1)
        np.testing.assert_allclose(normal_velocity, expected, rtol=1e-12, atol=1e-12)


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
    flow_conditions = _random_flow_velocity
    # Settings have nothing interesting besides the flow conditions
    settings = SolverSettings(
        flow_velocity=flow_conditions,
        model_settings=ModelSettings(
            vortex_cutoff=1e-6,
            vortex_far_approximation=1e-6,
            vortex_smallest_size=1e-6,
        ),
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
        vortex_cutoff=settings.model_settings.vortex_cutoff,
        vortex_far_approximation=settings.model_settings.vortex_far_approximation,
        vortex_smallest_size=settings.model_settings.vortex_smallest_size,
        quad_positions=pos.reshape(1, 4, 3),
        quad_circulations=state.circulation.mean(axis=0).reshape(-1),
        target_positions=tgt.reshape(1, 3),
    )
    # Get the flow velocity at the CP
    flow_vel = flow_conditions(time=target_time, positions=tgt)

    # Normal flow through CP should be (basically) zero for the no penetration condition
    assert np.isclose(np.dot(ind_vel + flow_vel, normal), 0)


def test_line_induction_matches_quad_induction() -> None:
    """Check line induction output matches quad induction by decomposing quads."""
    rng = np.random.default_rng(301)
    M = 5
    K = 8
    tol = 1e-8

    quad_positions = rng.random((M, 4, 3)) * 4 - 2
    quad_circulations = rng.random(M) * 2 - 1
    target_positions = rng.random((K, 3)) * 4 - 2

    # Map quads to individual line segments
    line_positions = np.empty((4 * M, 2, 3), dtype=np.double)
    line_circulations = np.empty((4 * M,), dtype=np.double)
    for i in range(M):
        element_pos = quad_positions[i]
        circ = quad_circulations[i]
        # Decompose the quad
        line_positions[4 * i + 0, 0, :] = element_pos[3]
        line_positions[4 * i + 0, 1, :] = element_pos[0]
        line_circulations[4 * i + 0] = circ

        line_positions[4 * i + 1, 0, :] = element_pos[0]
        line_positions[4 * i + 1, 1, :] = element_pos[1]
        line_circulations[4 * i + 1] = circ

        line_positions[4 * i + 2, 0, :] = element_pos[1]
        line_positions[4 * i + 2, 1, :] = element_pos[2]
        line_circulations[4 * i + 2] = circ

        line_positions[4 * i + 3, 0, :] = element_pos[2]
        line_positions[4 * i + 3, 1, :] = element_pos[3]
        line_circulations[4 * i + 3] = circ

    plane = TransformationPlane(origin=rng.random(3), normal=rng.random(3))

    for symmetry_plane in (None, plane):
        for n_threads in (1, 2):
            q_ind = quad_induction(
                vortex_cutoff=tol,
                vortex_far_approximation=tol,
                vortex_smallest_size=tol,
                quad_positions=quad_positions,
                quad_circulations=quad_circulations,
                target_positions=target_positions,
                symmetry_plane=symmetry_plane,
                n_threads=n_threads,
            )

            l_ind = line_induction(
                vortex_cutoff=tol,
                vortex_far_approximation=tol,
                vortex_smallest_size=tol,
                line_positions=line_positions,
                line_circulations=line_circulations,
                target_positions=target_positions,
                symmetry_plane=symmetry_plane,
                n_threads=n_threads,
            )

            np.testing.assert_allclose(l_ind, q_ind, rtol=1e-12, atol=1e-12)

            # Test using a pre-allocated out buffer
            out_buf = np.empty((K, 3), dtype=np.double)
            returned_buf = line_induction(
                vortex_cutoff=tol,
                vortex_far_approximation=tol,
                vortex_smallest_size=tol,
                line_positions=line_positions,
                line_circulations=line_circulations,
                target_positions=target_positions,
                symmetry_plane=symmetry_plane,
                out_velocity=out_buf,
                n_threads=n_threads,
            )
            assert returned_buf is out_buf
            np.testing.assert_allclose(out_buf, q_ind, rtol=1e-12, atol=1e-12)


def test_line_normal_induction_matches_quad_normal_induction() -> None:
    """Check line normal induction matches quad normal induction by decomposing quads."""
    rng = np.random.default_rng(302)
    M = 5
    K = 8
    tol = 1e-8

    quad_positions = rng.random((M, 4, 3)) * 4 - 2
    quad_circulations = rng.random(M) * 2 - 1
    target_positions = rng.random((K, 3)) * 4 - 2
    target_normals = rng.random((K, 3)) * 2 - 1
    target_normals /= np.linalg.norm(target_normals, axis=1, keepdims=True)

    # Map quads to individual line segments
    line_positions = np.empty((4 * M, 2, 3), dtype=np.double)
    line_circulations = np.empty((4 * M,), dtype=np.double)
    for i in range(M):
        element_pos = quad_positions[i]
        circ = quad_circulations[i]
        # Decompose the quad
        line_positions[4 * i + 0, 0, :] = element_pos[3]
        line_positions[4 * i + 0, 1, :] = element_pos[0]
        line_circulations[4 * i + 0] = circ

        line_positions[4 * i + 1, 0, :] = element_pos[0]
        line_positions[4 * i + 1, 1, :] = element_pos[1]
        line_circulations[4 * i + 1] = circ

        line_positions[4 * i + 2, 0, :] = element_pos[1]
        line_positions[4 * i + 2, 1, :] = element_pos[2]
        line_circulations[4 * i + 2] = circ

        line_positions[4 * i + 3, 0, :] = element_pos[2]
        line_positions[4 * i + 3, 1, :] = element_pos[3]
        line_circulations[4 * i + 3] = circ

    plane = TransformationPlane(origin=rng.random(3), normal=rng.random(3))

    for symmetry_plane in (None, plane):
        for n_threads in (1, 2):
            q_norm_ind = quad_normal_induction(
                vortex_cutoff=tol,
                vortex_far_approximation=tol,
                vortex_smallest_size=tol,
                quad_positions=quad_positions,
                quad_circulations=quad_circulations,
                target_positions=target_positions,
                target_normals=target_normals,
                symmetry_plane=symmetry_plane,
                n_threads=n_threads,
            )

            l_norm_ind = line_normal_induction(
                vortex_cutoff=tol,
                vortex_far_approximation=tol,
                vortex_smallest_size=tol,
                line_positions=line_positions,
                line_circulations=line_circulations,
                target_positions=target_positions,
                target_normals=target_normals,
                symmetry_plane=symmetry_plane,
                n_threads=n_threads,
            )

            np.testing.assert_allclose(l_norm_ind, q_norm_ind, rtol=1e-12, atol=1e-12)

            # Test using a pre-allocated out buffer
            out_buf = np.empty((K,), dtype=np.double)
            returned_buf = line_normal_induction(
                vortex_cutoff=tol,
                vortex_far_approximation=tol,
                vortex_smallest_size=tol,
                line_positions=line_positions,
                line_circulations=line_circulations,
                target_positions=target_positions,
                target_normals=target_normals,
                symmetry_plane=symmetry_plane,
                out_velocity=out_buf,
                n_threads=n_threads,
            )
            assert returned_buf is out_buf
            np.testing.assert_allclose(out_buf, q_norm_ind, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_line_circulation()
    test_induction_two_triangles_equal_to_quad()
    test_wake_induction_same_as_mesh()
    test_state_update()
    test_quad_induction_symmetry_plane_matches_mirrored_copy()
    test_quad_normal_induction_symmetry_plane_matches_mirrored_copy()
    test_quad_normal_induction_matches_velocity_projection()
    test_solver_system_updates_self_diagonal_on_global_motion()
    test_line_induction_matches_quad_induction()
    test_line_normal_induction_matches_quad_normal_induction()
