"""Tests related to the Mesh object."""

import numpy as np
import pytest
from pydust import Mesh


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
    np.random.seed(5463546)
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
            sm = merged.get_surface(n_surfaces)
            so = m.get_surface(i_s)
            for l1, l2 in zip(sm.lines, so.lines, strict=True):
                assert l1.begin == l2.begin + point_offset
                assert l1.end == l2.end + point_offset
            n_surfaces += 1

        for i_l in range(m.n_lines):
            lm = merged.get_line(n_lines)
            lo = m.get_line(i_l)
            assert lm.begin == lo.begin + point_offset
            assert lm.end == lo.end + point_offset
            n_lines += 1

        point_offset += m.n_points


def test_overlap_induction_matrix() -> None:
    """Check that induction matrix behaves nicely even on the lines."""
    msh = Mesh(4, [[0, 1, 2, 3]])
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float64)
    test_mat = msh.line_induction_matrix(1e-6, positions, positions)
    for i in range(4):
        assert all(test_mat[i, i, :] == (0, 0, 0))
