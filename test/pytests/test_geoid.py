"""Tests related to GeoID class."""

from pyvl import GeoID


def test_creation() -> None:
    """Check that creation and assignment work as intended."""
    id1 = GeoID(1, 4)
    assert id1.orientation
    id1.orientation = 0  # type: ignore
    assert not id1.orientation
    id1.index = 3
    assert id1.index == 3


def test_comparison() -> None:
    """Check that comparison operator works as intended."""
    assert GeoID(0, 1) == GeoID(0, 1)
    assert GeoID(42, 0) == GeoID(42, 0)
    assert GeoID(42, 1) != GeoID(42, 0)
    assert GeoID(42, 0) != GeoID(42, 1)
    assert GeoID(41, 0) != GeoID(42, 0)
    assert GeoID(42, 0) != GeoID(41, 0)
