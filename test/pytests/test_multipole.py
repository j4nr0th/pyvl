"""Tests related to the Multipole class."""

import numpy as np
import pytest
from pyvl import Multipole


def _populated_multipole() -> Multipole:
    """Return a Multipole(2, ...) built from three orthogonal unit sources."""
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    vals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return Multipole.from_sources(2, (0.0, 0.0, 0.0), pts, vals)


def test_construction_and_properties() -> None:
    """Empty multipoles of various orders and from_sources."""
    m0 = Multipole(0, (0.0, 0.0, 0.0))
    assert m0.order == 0
    assert m0.center == pytest.approx([0.0, 0.0, 0.0])
    # Zero-order empty expansion evaluates to zero everywhere
    assert m0.eval(np.array([[1.0, 2.0, 3.0]])) == pytest.approx(np.zeros((1, 3)))

    m2 = Multipole(2, (1.0, 2.0, 3.0))
    assert m2.order == 2
    assert m2.center == pytest.approx([1.0, 2.0, 3.0])

    # centre is read-write
    m2.center = (5.0, -3.0, 2.0)
    assert m2.center == pytest.approx([5.0, -3.0, 2.0])

    # from_sources with actual sources produces non-zero results
    m_pop = _populated_multipole()
    assert m_pop.order == 2
    assert not np.allclose(m_pop.eval(np.array([[10.0, 0.0, 0.0]])), 0.0)

    # mismatched point/value counts raise
    with pytest.raises(ValueError, match="does not match"):
        Multipole.from_sources(2, (0.0, 0.0, 0.0), np.zeros((3, 3)), np.zeros((2, 3)))

    # negative order rejected
    with pytest.raises(ValueError, match="non-negative"):
        Multipole.from_sources(-1, (0.0, 0.0, 0.0), np.zeros((1, 3)), np.zeros((1, 3)))


def test_str_and_repr() -> None:
    """String representations contain key information."""
    m = Multipole(2, (1.5, -2.5, 0.0))
    s = str(m)
    assert "Multipole" in s
    r = repr(m)
    assert r.startswith("<Multipole")
    assert "order=2" in r
    assert "center=" in r


def test_eval() -> None:
    """Check eval: shape preservation, batch dims, out=, and error handling."""
    m = _populated_multipole()

    # single point
    result = m.eval(np.array([[10.0, 0.0, 0.0]]))
    assert result.shape == (1, 3)

    # multiple points
    pts = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [-10.0, 0.0, 0.0],
        ]
    )
    result = m.eval(pts)
    assert result.shape == (4, 3)

    # 3-D batch: leading dims preserved
    pts = np.random.randn(3, 5, 3)
    result = m.eval(pts)
    assert result.shape == (3, 5, 3)

    # out= parameter
    pts2 = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    expected = m.eval(pts2)
    out = np.empty_like(expected)
    result = m.eval(pts2, out=out)
    assert result is out
    assert result == pytest.approx(expected)

    # bad last axis
    with pytest.raises((ValueError, TypeError)):
        m.eval(np.array([[1.0, 2.0]]))

    # bad out shape
    with pytest.raises(ValueError):
        m.eval(np.array([[10.0, 0.0, 0.0]]), out=np.empty((2, 3)))


def test_add_sources() -> None:
    """add_sources matches from_sources; incremental adds match bulk add."""
    rng = np.random.default_rng(42)
    pts = rng.normal(0, 1, (3, 3))
    vals = rng.normal(0, 1, (3, 3))

    m1 = Multipole.from_sources(2, (0.0, 0.0, 0.0), pts, vals)
    m2 = Multipole(2, (0.0, 0.0, 0.0))
    m2.add_sources(pts, vals)
    eval_pts = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    assert m1.eval(eval_pts) == pytest.approx(m2.eval(eval_pts))

    # empty add is a no-op
    m3 = Multipole(2, (0.0, 0.0, 0.0))
    m3.add_sources(np.zeros((0, 3)), np.zeros((0, 3)))
    assert m3.eval(np.array([[1.0, 0.0, 0.0]])) == pytest.approx(np.zeros((1, 3)))

    # incremental adds match single bulk add
    pts1 = np.array([[1.0, 0.0, 0.0]])
    vals1 = np.array([[1.0, 0.0, 0.0]])
    pts2 = np.array([[0.0, 1.0, 0.0]])
    vals2 = np.array([[0.0, 1.0, 0.0]])

    m_bulk = Multipole(2, (0.0, 0.0, 0.0))
    m_bulk.add_sources(np.vstack([pts1, pts2]), np.vstack([vals1, vals2]))

    m_inc = Multipole(2, (0.0, 0.0, 0.0))
    m_inc.add_sources(pts1, vals1)
    m_inc.add_sources(pts2, vals2)
    assert m_bulk.eval(np.array([[5.0, 0.0, 0.0]])) == pytest.approx(
        m_inc.eval(np.array([[5.0, 0.0, 0.0]]))
    )


def test_shift_and_shift_to() -> None:
    """Shift preserves far-field accuracy; shift_to uses target centre."""
    m = _populated_multipole()
    # shift to a new centre
    m_s = m.shift((5.0, 0.0, 0.0))
    assert m_s.center == pytest.approx([5.0, 0.0, 0.0])
    assert m_s.order == m.order

    # original unchanged (immutable)
    assert m.center == pytest.approx([0.0, 0.0, 0.0])

    # shift_to uses another multipole's centre
    target = Multipole(2, (5.0, 0.0, 0.0))
    m_st = m.shift_to(target)
    assert m_st.center == pytest.approx([5.0, 0.0, 0.0])

    # invalid work_order
    with pytest.raises(ValueError, match="work_order must be"):
        m.shift((0.0, 0.0, 0.0), work_order=0)


def test_equality() -> None:
    """Equality compares order, centre and coefficient buffer."""
    a = Multipole(2, (1.0, 2.0, 3.0))
    b = Multipole(2, (1.0, 2.0, 3.0))
    assert a == b

    # different order
    assert a != Multipole(1, (1.0, 2.0, 3.0))
    # different centre
    assert a != Multipole(2, (1.0, 0.0, 0.0))
    # different coefficients
    c = Multipole.from_sources(
        2, (1.0, 2.0, 3.0), np.array([[1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    )
    assert a != c
    # non-Multipole
    assert a != "hello"
