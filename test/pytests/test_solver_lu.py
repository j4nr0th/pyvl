"""Test the solver system works."""

import numpy as np
import numpy.typing as npt
import pytest
from pyvl.cvl import Mesh, ReferenceFrame
from pyvl.geometry import Geometry
from pyvl.solver import SolverSystem


def _compute_normal_rhs(
    solver: SolverSystem,
    vtol: float,
    real_circulations: dict[str, npt.NDArray[np.double]],
    time: float = 0.0,
) -> dict[str, npt.NDArray[np.double]]:
    """Compute the RHS using the solver's own matrix assembly path.

    For self-induction, uses the solver's self_induction_diags (local coords).
    For cross-induction, uses compute_induction_matrix (global coords at given time)
    on a fresh output to avoid corrupting the LU state.
    """
    y = {label: np.zeros_like(real_circulations[label]) for label in real_circulations}
    for target_name in solver.part_order:
        target_geo = solver.geometry[target_name]
        for source_name in solver.part_order:
            source_geo = solver.geometry[source_name]
            if source_name == target_name:
                nmat = solver.self_induction_diags[source_name]
            else:
                source_pos = source_geo.reference_frame.to_global_position(
                    source_geo.positions, time=time
                )
                target_cpts = target_geo.reference_frame.to_global_position(
                    target_geo.centers, time=time
                )
                target_normals = target_geo.reference_frame.to_global_vector(
                    target_geo.normals, time=time
                )
                nmat = source_geo.msh.induction_matrix3(
                    tol=vtol,
                    positions=source_pos,
                    control_points=target_cpts,
                    normals=target_normals,
                )
            y[target_name] += nmat @ real_circulations[source_name]
    return y


def test_solver_system_lu_caching():
    """Verify that SolverSystem caches LU and updates when geometries move."""
    # Create two geometries
    points1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.double)
    connectivity1 = [np.array([0, 1, 2, 3], dtype=np.uint32)]
    mesh1 = Mesh(len(points1), connectivity1)
    rf1 = ReferenceFrame()
    geo1 = Geometry("geo1", rf1, mesh1, points1)

    points2 = np.array([[2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]], dtype=np.double)
    connectivity2 = [np.array([0, 1, 2, 3], dtype=np.uint32)]
    mesh2 = Mesh(len(points2), connectivity2)
    rf2 = ReferenceFrame()
    geo2 = Geometry("geo2", rf2, mesh2, points2)

    # Initialize SolverSystem
    SolverSystem(1e-6, geo1, geo2)

    # TODO: check the system works with


def test_solver_system_inverse_consistency():
    """Verify consistency of solve_inverse with forward multiplication."""
    rng = np.random.default_rng(3935)
    vtol = 1e-15
    # Create simple geometries
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.double)
    connectivity = [np.array([0, 1, 2, 3], dtype=np.uint32)]

    geos = []
    for i in range(2):
        mesh = Mesh(len(points), connectivity)
        # Re-initialize rf using the constructor which takes
        # (offset, theta, velocity, rotation, parent)
        rf = ReferenceFrame(offset=rng.uniform(-10, 10, 3))
        geos.append(Geometry(f"geo{i}", rf, mesh, points))

    solver = SolverSystem(vtol, *geos)

    # Prepare random true solution
    x_true = {g.label: rng.uniform(-10, +10, g.msh.n_surfaces) for g in geos}

    # Compute y = A * x_true manually using the stored induction matrices
    y = _compute_normal_rhs(solver, vtol, x_true)

    # Use solver to solve A * u = y
    u = {k: v.copy() for k, v in y.items()}
    solver.solve_inverse(u)

    # Verify u is close to x_true
    for label in u:
        assert u[label] == pytest.approx(x_true[label], rel=1e-10)


def test_solver_system_inverse_complex_moving():
    """Verify consistency of solve_inverse with multiple moving geometries."""
    rng = np.random.default_rng(3935)
    vtol = 1e-15

    quad_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.double)
    quad_conn = [np.array([0, 1, 2, 3], dtype=np.uint32)]

    def make_time_varying_frame(rng):
        offset0 = rng.uniform(-10, 10, 3)
        offset1 = rng.uniform(-10, 10, 3)
        theta0 = rng.uniform(-np.pi / 4, np.pi / 4, 3)
        theta1 = rng.uniform(-np.pi / 4, np.pi / 4, 3)

        def offset_fn(t):
            return offset0 + (offset1 - offset0) * t

        def theta_fn(t):
            return theta0 + (theta1 - theta0) * t

        return ReferenceFrame(offset=offset_fn, theta=theta_fn)

    geos = [
        Geometry(
            "geo0",
            make_time_varying_frame(rng),
            Mesh(len(quad_points), quad_conn),
            quad_points,
        ),
        Geometry(
            "geo1",
            make_time_varying_frame(rng),
            Mesh(len(quad_points), quad_conn),
            quad_points,
        ),
        Geometry(
            "geo2",
            make_time_varying_frame(rng),
            Mesh(len(quad_points), quad_conn),
            quad_points,
        ),
    ]

    solver = SolverSystem(vtol, *geos)

    solver.update_induction_matrices(t_start=0.0, t_end=1.0, tol=vtol)

    x_true = {g.label: rng.uniform(-10, +10, g.msh.n_surfaces) for g in geos}
    y = _compute_normal_rhs(solver, vtol, x_true, time=1.0)

    u = {k: v.copy() for k, v in y.items()}
    solver.solve_inverse(u)

    for label in u:
        assert u[label] == pytest.approx(x_true[label], rel=1e-10)


def test_solver_system_multi_move_cycles():
    """Verify inverse consistency through multiple move/update cycles."""
    rng = np.random.default_rng(42)
    vtol = 1e-15

    quad_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.double)
    quad_conn = [np.array([0, 1, 2, 3], dtype=np.uint32)]

    tri_points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.double)
    tri_conn = [np.array([0, 1, 2], dtype=np.uint32)]

    def make_moving_frame(rng):
        offset0 = rng.uniform(-10, 10, 3)
        offset1 = rng.uniform(-10, 10, 3)
        theta0 = rng.uniform(-np.pi / 4, np.pi / 4, 3)
        theta1 = rng.uniform(-np.pi / 4, np.pi / 4, 3)

        def offset_fn(t):
            return offset0 + (offset1 - offset0) * t

        def theta_fn(t):
            return theta0 + (theta1 - theta0) * t

        return ReferenceFrame(offset=offset_fn, theta=theta_fn)

    geos = []
    for i in range(3):
        mesh = Mesh(len(quad_points), quad_conn)
        geos.append(Geometry(f"quad{i}", make_moving_frame(rng), mesh, quad_points))

    mesh = Mesh(len(tri_points), tri_conn)
    geos.append(Geometry("tri0", make_moving_frame(rng), mesh, tri_points))

    solver = SolverSystem(vtol, *geos)

    for move_cycle in range(4):
        # Build new reference frames with fresh target positions
        for g in geos:
            old_rf = g.reference_frame
            offset_b = rng.uniform(-10, 10, 3)
            theta_b = rng.uniform(-np.pi / 4, np.pi / 4, 3)

            def make_new_frame(old_rf=old_rf, offset_b=offset_b, theta_b=theta_b):
                offset_a = old_rf.offset_at(0.0)
                theta_a = old_rf.angles_at(0.0)

                def offset_fn(t):
                    return offset_a + (offset_b - offset_a) * t

                def theta_fn(t):
                    return theta_a + (theta_b - theta_a) * t

                return ReferenceFrame(offset=offset_fn, theta=theta_fn)

            object.__setattr__(g, "reference_frame", make_new_frame())

        t_start = float(move_cycle) + 1.0
        t_end = t_start + 1.0
        solver.update_induction_matrices(t_start=t_start, t_end=t_end, tol=vtol)

        x_true = {g.label: rng.uniform(-5, 5, g.msh.n_surfaces) for g in geos}
        y = _compute_normal_rhs(solver, vtol, x_true, time=t_end)

        u = {k: v.copy() for k, v in y.items()}
        solver.solve_inverse(u)

        for label in u:
            assert u[label] == pytest.approx(x_true[label], rel=1e-10), (
                f"Failed at move cycle {move_cycle} for {label}"
            )


def test_solver_system_mixed_motion_groups():
    """Verify inverse when some geometries stay fixed and others move."""
    rng = np.random.default_rng(12345)
    vtol = 1e-15

    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.double)
    conn = [np.array([0, 1, 2, 3], dtype=np.uint32)]

    fixed_rf = ReferenceFrame(offset=np.array([0.0, 0.0, 0.0]))
    geos = [
        Geometry("fixed1", fixed_rf, Mesh(len(points), conn), points),
        Geometry("fixed2", fixed_rf, Mesh(len(points), conn), points + [2, 0, 0]),
    ]

    def make_moving_frame(rng):
        offset0 = rng.uniform(-1, 1, 3)
        offset1 = rng.uniform(-1, 1, 3)

        def offset_fn(t):
            return offset0 + (offset1 - offset0) * t

        return ReferenceFrame(offset=offset_fn)

    geos.append(
        Geometry("moving1", make_moving_frame(rng), Mesh(len(points), conn), points)
    )

    solver = SolverSystem(vtol, *geos)

    for move_cycle in range(3):
        # Replace the moving frame with a new time-varying one
        t_start = float(move_cycle) + 1.0
        t_end = t_start + 1.0
        solver.update_induction_matrices(t_start=t_start, t_end=t_end, tol=vtol)

        x_true = {g.label: rng.uniform(-5, 5, g.msh.n_surfaces) for g in geos}
        y = _compute_normal_rhs(solver, vtol, x_true, time=t_end)

        u = {k: v.copy() for k, v in y.items()}
        solver.solve_inverse(u)

        for label in u:
            assert u[label] == pytest.approx(x_true[label]), (
                f"Failed at move cycle {move_cycle} for {label}"
            )


if __name__ == "__main__":
    test_solver_system_inverse_consistency()
    test_solver_system_inverse_complex_moving()
    test_solver_system_multi_move_cycles()
    test_solver_system_mixed_motion_groups()
