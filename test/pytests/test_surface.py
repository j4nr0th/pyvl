"""Tests for the surface objects."""

from pydust import Line, Surface


def test_creation():
    """Test that it can be created as expected and getters/setters work."""
    s = Surface([Line(0, 1), Line(1, 3), Line(3, 5), Line(5, 0)])
    s2 = Surface(tuple(ln for ln in s.lines))
    assert s.lines == s2.lines
    assert s.n_lines == s2.n_lines


def test_comparison():
    """Test that surface shows up as same if lines are cycled."""
    s = Surface([Line(0, 1), Line(1, 3), Line(3, 5), Line(5, 0)])
    for i in range(s.n_lines):
        s2 = Surface(s.lines[i:] + s.lines[:i])
        assert s == s2


def test_bad_construction():
    """Check that construction fails with unconnected lines."""
    caught = False
    try:
        # First point does not connect
        s = Surface([Line(3, 1), Line(1, 3), Line(3, 5), Line(5, 0)])
        del s
    except ValueError:
        caught = True
    assert caught
    caught = False
    try:
        # Middle line has a  point that does not connect
        s = Surface([Line(0, 1), Line(1, 9), Line(3, 5), Line(5, 0)])
        del s
    except ValueError:
        caught = True
    assert caught
    caught = False
    try:
        # Last has a  point that does not connect
        s = Surface([Line(0, 1), Line(1, 9), Line(3, 5), Line(5, 100)])
        del s
    except ValueError:
        caught = True
    assert caught
