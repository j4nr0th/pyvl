"""Tests related to the Mesh object."""

import numpy as np
import pytest
from pyvl import Mesh, TransformationPlane


def test_mesh_construction():
    """Check that constructor and getters/setters works as intended."""
    elements = [[0, 1, 2, 3], [2, 3, 4], [0, 1, 4], [2, 4, 5]]
    msh = Mesh(max(max(e) for e in elements), elements)
    flat_elements = np.array(sum(elements, start=[]))
    element_count, connectivity = msh.to_element_connectivity()
    assert np.all(element_count == [len(e) for e in elements])
    assert all(connectivity == flat_elements)


def test_mesh_surface_normals():
    """Check that surface normals are all really unit normals."""
    positions = np.array(
        [
            [0, 2, 0.5],
            [0.4, 2, 0.4],
            [4, 0.1, 0.2],
            [3, -0.1, 0.2],
            [2, 1, 0.2],
            [2, -1, 3],
        ]
    )
    elements = [[0, 1, 2, 3], [2, 3, 4], [0, 1, 4], [2, 4, 5]]
    msh = Mesh(positions.shape[0], elements)
    normals = msh.surface_normal(positions)
    # Unit length
    assert 1 == pytest.approx(np.linalg.norm(normals, axis=1))
    # For triangle elements, these should be perpendicular to all lines
    # other elements are more ticky, since if they're not planar, the normal
    # will be computed as weighted average.
    for n, e in zip(normals, elements):
        if len(e) != 3:
            continue
        r0 = positions[e[0], :] - positions[e[2], :]
        r1 = positions[e[1], :] - positions[e[0], :]
        r2 = positions[e[2], :] - positions[e[1], :]

        assert 0 == pytest.approx(np.dot(r1, n))
        assert 0 == pytest.approx(np.dot(r2, n))
        assert 0 == pytest.approx(np.dot(r0, n))


def test_mesh_surface_centers():
    """Check that surface centers are all correct."""
    positions = np.array(
        [
            [0, 2, 0.5],
            [0.4, 2, 0.4],
            [4, 0.1, 0.2],
            [3, -0.1, 0.2],
            [2, 1, 0.2],
            [2, -1, 3],
        ]
    )
    elements = [[0, 1, 2, 3], [2, 3, 4], [0, 1, 4], [2, 4, 5]]
    msh = Mesh(positions.shape[0], elements)
    centers = msh.surface_average_vec3(positions)
    for c, e in zip(centers, elements):
        v = np.mean(positions[e, :], axis=0)

        assert v == pytest.approx(c)


def test_mesh_merge():
    """Check that multiple meshes can be successfully merged."""
    msh1 = Mesh(6, [[0, 1, 2, 3], [2, 3, 4], [0, 1, 4], [2, 4, 5]])
    msh2 = Mesh(6, [[2, 1, 0], [1, 2, 3], [4, 1, 5]])
    msh3 = Mesh(6, [[4, 1, 0], [5, 2, 4]])

    merged = Mesh.merge_meshes(msh1, msh2, msh3)

    assert merged.n_surfaces == msh1.n_surfaces + msh2.n_surfaces + msh3.n_surfaces
    assert merged.n_lines == msh1.n_lines + msh2.n_lines + msh3.n_lines
    assert merged.n_points == msh1.n_points + msh2.n_points + msh3.n_points
    n_lines = 0
    n_surfaces = 0
    point_offset = 0
    for m in (msh1, msh2, msh3):
        for i_s in range(m.n_surfaces):
            sm = merged.get_surface_lines(n_surfaces)
            so = m.get_surface_lines(i_s)
            for l1, l2 in zip(sm, so, strict=True):
                assert l1.index == l2.index + n_lines
                assert l1.orientation == l2.orientation
            n_surfaces += 1

        for i_l in range(m.n_lines):
            lm = merged.get_line_points(n_lines)
            lo = m.get_line_points(i_l)
            assert lm[0] == lo[0] + point_offset
            assert lm[1] == lo[1] + point_offset
            n_lines += 1

        point_offset += m.n_points


def test_mesh_accepts_symmetry_plane() -> None:
    """Check that mesh induction with symmetry matches an explicit mirrored copy."""
    rng = np.random.default_rng(205)
    msh = Mesh(4, [[0, 1, 2, 3]])
    positions = rng.random((4, 3))
    plane = TransformationPlane(origin=rng.random(3), normal=rng.random(3))
    mirrored_positions = plane.reflect(positions)
    line_circulation = rng.random(4)
    control_points = rng.random((5, 3))

    base = msh.induction_velocity(
        1e-8,
        positions,
        control_points,
        line_circulation,
    )
    mirrored = msh.induction_velocity(
        1e-8,
        mirrored_positions,
        control_points,
        -line_circulation,
    )
    with_symmetry = msh.induction_velocity(
        1e-8,
        positions,
        control_points,
        line_circulation,
        symmetry_plane=plane,
    )

    expected = base + mirrored

    np.testing.assert_allclose(with_symmetry, expected, rtol=1e-12, atol=1e-12)

    # Now check that putting the control points on the plane results in
    # no normal induction
    control_points_on_plane = (plane.reflect(control_points) + control_points) / 2
    with_symmetry = msh.induction_velocity(
        1e-8,
        positions,
        control_points_on_plane,
        line_circulation,
        symmetry_plane=plane,
    )
    np.testing.assert_allclose(
        with_symmetry @ plane.normal(), 0.0, rtol=1e-12, atol=1e-12
    )


def test_induction_matrix3_matches_velocity_projection() -> None:
    """Check matrix3 matches velocity projected onto target normals."""
    rng = np.random.default_rng(206)
    msh = Mesh(
        7,
        [
            [0, 1, 2, 3],
            [0, 3, 4],
            [2, 5, 6],
        ],
    )
    positions = rng.random((msh.n_points, 3)) * 4 - 2
    surface_circulations = rng.random(msh.n_surfaces) * 2 - 1
    control_points = rng.random((8, 3)) * 4 - 2
    target_normals = rng.random((8, 3)) * 2 - 1
    target_normals /= np.linalg.norm(target_normals, axis=1, keepdims=True)
    plane = TransformationPlane(origin=rng.random(3), normal=rng.random(3))

    line_circulations = msh.line_circulations(surface_circulations)

    for symmetry_plane in (None, plane):
        matrix3 = msh.induction_matrix3(
            1e-8,
            positions,
            control_points,
            target_normals,
            symmetry_plane=symmetry_plane,
        )
        velocity = msh.induction_velocity(
            1e-8,
            positions,
            control_points,
            line_circulations,
            symmetry_plane=symmetry_plane,
        )

        expected = np.sum(velocity * target_normals, axis=1)
        np.testing.assert_allclose(
            matrix3 @ surface_circulations, expected, rtol=1e-12, atol=1e-12
        )


if __name__ == "__main__":
    test_mesh_construction()
    test_mesh_surface_normals()
    test_mesh_surface_centers()
    test_mesh_merge()
    test_mesh_accepts_symmetry_plane()
    test_induction_matrix3_matches_velocity_projection()
