"""Tests related to the TransformationPlane class."""

import numpy as np
import pytest
from pyvl import TransformationPlane


def test_transformation_plane_constant_fields() -> None:
    """Check that constant plane fields are returned as expected."""
    plane = TransformationPlane(origin=(1.0, 2.0, 3.0), normal=(0.0, 0.0, 2.0))

    assert plane.origin() == pytest.approx([1.0, 2.0, 3.0])
    assert plane.normal() == pytest.approx([0.0, 0.0, 1.0])


def test_transformation_plane_callable_fields() -> None:
    """Check that callable plane fields are evaluated at the requested time."""

    def origin_func(t: float) -> tuple[float, float, float]:
        return (t, 2.0 * t, 3.0 * t)

    def normal_func(t: float) -> tuple[float, float, float]:
        return (0.0, 0.0, 1.0 + t)

    plane = TransformationPlane(origin=origin_func, normal=normal_func)

    assert plane.origin(2.5) == pytest.approx([2.5, 5.0, 7.5])
    assert plane.normal(2.5) == pytest.approx([0.0, 0.0, 1.0])


def test_transformation_plane_reflect() -> None:
    """Check that reflecting points across the plane works."""
    plane = TransformationPlane(origin=(1.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    points = np.array([[2.0, -1.0, 4.0], [4.5, 8.0, -3.0]])

    reflected = plane.reflect(points)

    np.testing.assert_allclose(reflected, [[0.0, -1.0, 4.0], [-2.5, 8.0, -3.0]])
