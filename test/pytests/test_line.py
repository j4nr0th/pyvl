"""Tests for Line object."""

from pyvl import Line


def test_creation():
    """Test that creation and getters/setters work."""
    ln = Line(0, 2)
    assert ln.begin == 0
    assert ln.end == 2
    ln.begin = 3
    assert ln.begin == 3


def test_comparison():
    """Test that comparison operator works."""
    assert Line(0, 2) == Line(0, 2)
    assert Line(0, 2) != Line(0, 1)
    assert Line(0, 2) != Line(1, 2)
    assert Line(0, 2) != Line(2, 0)
