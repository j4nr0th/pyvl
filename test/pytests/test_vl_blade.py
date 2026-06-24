"""Tests for the VLBlade mesh generator."""

import numpy as np
from pyvl import Mesh
from pyvl.meshing.vl_blade import VLBlade


def test_make_quad2d_plane_basic_topology() -> None:
    """Check that the quad plane mesh has the expected basic topology."""
    mesh = Mesh.make_quad2d_plane(2, 3)

    assert mesh.n_points == 12
    assert mesh.n_lines == 17
    assert mesh.n_surfaces == 6

    assert mesh.get_line_points(0) == (0, 1)
    assert mesh.get_line_points(7) == (10, 11)
    assert mesh.get_line_points(8) == (0, 3)
    assert mesh.get_line_points(16) == (8, 11)

    first_surface = mesh.get_surface_lines(0)
    assert [(line.index, line.orientation) for line in first_surface] == [
        (0, False),
        (9, False),
        (2, True),
        (8, True),
    ]

    last_surface = mesh.get_surface_lines(mesh.n_surfaces - 1)
    assert [(line.index, line.orientation) for line in last_surface] == [
        (5, False),
        (16, False),
        (7, True),
        (15, True),
    ]

    n_per_element, flattened = mesh.to_element_connectivity()
    assert np.all(n_per_element == 4)
    assert flattened.size == 24


def test_mesh_geometry_places_le_at_start_for_zero_reference_chord_fraction() -> None:
    """Check that the first point of each section is the LE when ref chord is zero."""
    blade = VLBlade(
        reference_line=lambda s: np.array([0.0, s, 0.0]),
        reference_chord_fraction=0.0,
        chord_distribution=2.0,
        twist_distribution=0.0,
        camber_distribution=None,
    )

    spanwise_positions = np.array([0.0, 0.5, 1.0])
    chordwise_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    geo = blade.mesh_geometry(spanwise_positions, chordwise_positions)
    assert geo.msh.n_points == geo.positions.shape[0]
    assert geo.msh.n_points == spanwise_positions.size * chordwise_positions.size
    sections = geo.positions.reshape(chordwise_positions.size, spanwise_positions.size, 3)

    for span_index, span in enumerate(spanwise_positions):
        section = sections[:, span_index, :]
        np.testing.assert_allclose(section[0], [0.0, span, 0.0])
        np.testing.assert_allclose(section[-1], [2.0, span, 0.0])
        assert np.all(np.diff(section[:, 0]) > 0)

    assert geo.msh.get_line_points(0) == (0, 1)
    assert geo.msh.get_line_points(21) == (11, 14)


def test_mesh_geometry_places_te_at_end_for_unit_reference_chord_fraction() -> None:
    """Check that the trailing edge lands on the reference line for reference fraction."""
    blade = VLBlade(
        reference_line=lambda s: np.array([0.0, s, 0.0]),
        reference_chord_fraction=1.0,
        chord_distribution=2.0,
        twist_distribution=0.0,
        camber_distribution=None,
    )

    spanwise_positions = np.array([0.0, 0.5, 1.0])
    chordwise_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    geo = blade.mesh_geometry(spanwise_positions, chordwise_positions)
    assert geo.msh.n_lines == 22
    assert geo.msh.n_surfaces == 8
    sections = geo.positions.reshape(chordwise_positions.size, spanwise_positions.size, 3)

    for span_index, span in enumerate(spanwise_positions):
        section = sections[:, span_index, :]
        np.testing.assert_allclose(section[0], [-2.0, span, 0.0])
        np.testing.assert_allclose(section[-1], [0.0, span, 0.0])
        assert np.all(np.diff(section[:, 0]) > 0)

    assert geo.msh.get_line_points(0) == (0, 1)
    assert geo.msh.get_line_points(21) == (11, 14)


def test_mesh_geometry_mesh_connectivity_matches_grid() -> None:
    """Check that the generated mesh connectivity matches the section grid."""
    blade = VLBlade(
        reference_line=lambda s: np.array([0.0, s, 0.0]),
        reference_chord_fraction=0.0,
        chord_distribution=1.0,
        twist_distribution=0.0,
        camber_distribution=None,
    )

    spanwise_positions = np.array([0.0, 0.5, 1.0])
    chordwise_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    geo = blade.mesh_geometry(spanwise_positions, chordwise_positions)
    mesh = geo.msh

    assert mesh.n_points == 15
    assert mesh.n_lines == 22
    assert mesh.n_surfaces == 8

    assert mesh.get_line_points(0) == (0, 1)
    assert mesh.get_line_points(1) == (1, 2)
    assert mesh.get_line_points(7) == (10, 11)
    assert mesh.get_line_points(8) == (12, 13)
    assert mesh.get_line_points(21) == (11, 14)

    first_surface = mesh.get_surface_lines(0)
    assert [(line.index, line.orientation) for line in first_surface] == [
        (0, False),
        (11, False),
        (2, True),
        (10, True),
    ]

    last_surface = mesh.get_surface_lines(mesh.n_surfaces - 1)
    assert [(line.index, line.orientation) for line in last_surface] == [
        (7, False),
        (21, False),
        (9, True),
        (20, True),
    ]
