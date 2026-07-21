"""Tests related to the BarnesHutTree class."""

import numpy as np
import pytest
from pyvl import BarnesHutTree


def _random_sources(n=200, seed=42):
    """Return (coords, values) with *n* sources spread across a unit cube."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1.0, 1.0, (n, 3))
    values = rng.uniform(-1.0, 1.0, (n, 3))
    return coords, values


def test_build_and_basic_properties():
    """Build a tree and verify all basic properties are populated."""
    coords, values = _random_sources(200)
    tree = BarnesHutTree.build(coords, values, order=4)

    assert tree.n_sources == 200
    assert tree.n_nodes > 0
    assert tree.n_internal > 0
    assert tree.n_multipole_leaves > 0 or tree.n_particle_leaves > 0
    assert (
        tree.n_nodes == tree.n_internal + tree.n_multipole_leaves + tree.n_particle_leaves
    )
    assert tree.max_depth > 0
    assert tree.max_depth <= 20
    assert tree.memory_bytes > 0
    assert tree.order == 4
    assert tree.critical_particle_count == 4
    assert tree.max_depth_setting == 20


def test_build_default_order():
    """Default order=4 produces a valid tree."""
    coords, values = _random_sources(100)
    tree = BarnesHutTree.build(coords, values)
    assert tree.order == 4
    assert tree.n_sources == 100


def test_eval_shape_preservation():
    """Eval preserves leading dimensions of the target array."""
    coords, values = _random_sources(200)
    tree = BarnesHutTree.build(coords, values, order=4)

    # (N, 3) -> (N, 3)
    pts = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [-5.0, 0.0, 0.0]])
    result = tree.eval(pts)
    assert result.shape == (4, 3)

    # 3-D batch: (3, 5, 3) -> (3, 5, 3)
    pts3d = np.random.randn(3, 5, 3)
    result3d = tree.eval(pts3d)
    assert result3d.shape == (3, 5, 3)


def test_eval_out_parameter():
    """Output array can be provided via the out= parameter."""
    coords, values = _random_sources(200)
    tree = BarnesHutTree.build(coords, values, order=4)

    pts = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    expected = tree.eval(pts)
    out = np.empty_like(expected)
    result = tree.eval(pts, out=out)
    assert result is out
    assert result == pytest.approx(expected)


def test_eval_error_paths():
    """Bad inputs to eval raise appropriate exceptions."""
    coords, values = _random_sources(100)
    tree = BarnesHutTree.build(coords, values, order=4)

    # Bad last axis
    with pytest.raises((ValueError, TypeError)):
        tree.eval(np.array([[1.0, 2.0]]))

    # Bad out shape
    with pytest.raises(ValueError):
        tree.eval(np.array([[10.0, 0.0, 0.0]]), out=np.empty((2, 3)))

    # Wrong dtype for out
    with pytest.raises((ValueError, TypeError)):
        tree.eval(np.array([[10.0, 0.0, 0.0]]), out=np.empty((1, 3), dtype=np.float32))


def test_eval_unbuilt_tree():
    """Calling eval on an unbuilt tree raises RuntimeError."""
    tree = BarnesHutTree(order=4)
    with pytest.raises(RuntimeError, match="not been built"):
        tree.eval(np.array([[1.0, 0.0, 0.0]]))


def test_determinism():
    """Same seed produces identical tree shape."""
    coords_a, values_a = _random_sources(200, seed=123)
    coords_b, values_b = _random_sources(200, seed=123)

    tree_a = BarnesHutTree.build(coords_a, values_a, order=4)
    tree_b = BarnesHutTree.build(coords_b, values_b, order=4)

    assert tree_a.n_nodes == tree_b.n_nodes
    assert tree_a.n_internal == tree_b.n_internal
    assert tree_a.n_multipole_leaves == tree_b.n_multipole_leaves
    assert tree_a.n_particle_leaves == tree_b.n_particle_leaves
    assert tree_a.max_depth == tree_b.max_depth


def test_str_and_repr():
    """String representations contain key information."""
    coords, values = _random_sources(200)
    tree = BarnesHutTree.build(coords, values, order=4)

    s = str(tree)
    assert "BarnesHutTree" in s
    assert "order=4" in s
    assert "n_sources=200" in s

    r = repr(tree)
    assert r.startswith("<BarnesHutTree")
    assert "order=4" in r
    assert "n_sources=200" in r


def test_unbuilt_str_repr():
    """Unbuilt tree string representations mention unbuilt."""
    tree = BarnesHutTree(order=4)
    s = str(tree)
    assert "unbuilt" in s
    r = repr(tree)
    assert "unbuilt" in r


def test_negative_order_rejected():
    """Negative order is rejected."""
    coords, values = _random_sources(10)
    with pytest.raises(ValueError):
        BarnesHutTree.build(coords, values, order=0)

    # Also rejected at construction
    with pytest.raises(ValueError):
        BarnesHutTree(order=0)


def test_mismatched_source_counts():
    """Mismatched coords/values counts raise ValueError."""
    coords = np.ones((10, 3))
    values = np.ones((5, 3))
    with pytest.raises(ValueError, match="does not match"):
        BarnesHutTree.build(coords, values, order=4)


def test_critical_particle_count():
    """critical_particle_count can be set explicitly."""
    coords, values = _random_sources(200)
    tree = BarnesHutTree.build(coords, values, order=4, critical_particle_count=8)
    assert tree.critical_particle_count == 8


def test_n_threads_propagation():
    """n_threads passed to build is used as default for eval."""
    coords, values = _random_sources(200)
    tree = BarnesHutTree.build(coords, values, order=4, n_threads=2)
    pts = np.array([[5.0, 0.0, 0.0]])
    result = tree.eval(pts)
    assert result.shape == (1, 3)


def test_theta_parameter():
    """Check theta > 0 (opening-angle mode) does not crash."""
    coords, values = _random_sources(200)
    tree = BarnesHutTree.build(coords, values, order=4)
    pts = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    result = tree.eval(pts, theta=0.5)
    assert result.shape == (3, 3)
    assert np.all(np.isfinite(result))
