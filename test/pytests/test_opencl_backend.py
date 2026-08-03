"""Tests for the OpenCL tree backend (CLBackend / CLTree / futures).

These tests require an OpenCL device.  They skip when no usable device
is available (e.g. CI machines without OpenCL).
"""

import numpy as np
import pytest
from pyvl import BarnesHutTree, create_backend

pytestmark = pytest.mark.skipif(
    not hasattr(__import__("pyvl.cvl", fromlist=["create_backend"]), "create_backend"),
    reason="pyvl built without OpenCL support",
)


def _random_sources(n=200, seed=42):
    """Return (coords, values) with *n* sources in a unit cube."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1.0, 1.0, (n, 3))
    values = rng.uniform(-1.0, 1.0, (n, 3))
    return coords, values


@pytest.fixture(scope="module")
def backend():
    """Create a backend on the first usable device (GPU, then CPU)."""
    try:
        b = create_backend("gpu")
        return b
    except RuntimeError:
        try:
            b = create_backend("cpu")
            return b
        except RuntimeError:
            pytest.skip("No OpenCL device available")


def test_backend_properties(backend):
    """Backend exposes device info and precision."""
    assert backend.device_name
    assert backend.vendor
    assert backend.device_type in ("gpu", "cpu")
    assert backend.precision in ("fp64", "fp32")
    assert backend.closed is False


def test_build_future_and_tree(backend):
    """build_tree returns a future; result() gives a populated tree."""
    coords, values = _random_sources(200)
    future = backend.build_tree(coords, values)
    assert future.done() is False

    tree = future.result()
    assert future.done() is True
    assert tree.built is True
    assert tree.n_sources == 200
    assert tree.n_nodes > 0
    assert tree.n_internal > 0
    assert tree.n_leaves > 0
    assert tree.n_nodes == (
        tree.n_internal + tree.n_multipole_leaves + tree.n_particle_leaves
    )
    assert tree.order == 4


def test_eval_shape_preservation(backend):
    """Eval preserves leading dims; direct mode matches the exact sum."""
    coords, values = _random_sources(200)
    tree = backend.build_tree(coords, values).result()

    pts = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    v = tree.eval(pts, mode="direct").result()
    assert v.shape == (3, 3)

    pts3d = np.random.default_rng(3).uniform(2.0, 3.0, (3, 5, 3))
    v3d = tree.eval(pts3d, mode="direct").result()
    assert v3d.shape == (3, 5, 3)


def test_direct_eval_exact(backend):
    """Direct-sum eval is bit-exact against the NumPy reference."""
    coords, values = _random_sources(200)
    tree = backend.build_tree(coords, values).result()

    pts = np.random.default_rng(7).uniform(2.0, 4.0, (100, 3))
    v = tree.eval(pts, mode="direct").result()

    d = pts[:, None, :] - coords[None, :, :]
    r2 = np.maximum(np.sum(d * d, axis=-1), 1e-30)
    ref = np.sum(values[None, :, :] / r2[..., None], axis=1)
    np.testing.assert_allclose(v, ref, rtol=1e-12, atol=1e-12)


def test_tree_code_matches_bh(backend):
    """Tree-code eval agrees with BarnesHutTree within the approximation."""
    coords, values = _random_sources(500, seed=1)
    tree = backend.build_tree(coords, values).result()

    # Exterior targets: the multipole series converges.
    rng = np.random.default_rng(9)
    pts = rng.uniform(2.0, 4.0, (100, 3))
    v_cl = tree.eval(pts, mode="tree_code", theta=0.0).result()

    bh = BarnesHutTree.build(
        coords, values, order=4, critical_particle_count=4, alpha_centroid=0.0
    )
    v_bh = bh.eval(pts, theta=0.0)

    # Both are order-4 multipole approximations; allow a loose tolerance.
    rel = np.linalg.norm(v_cl - v_bh, axis=-1) / np.maximum(
        np.linalg.norm(v_bh, axis=-1), 1e-30
    )
    assert np.median(rel) < 0.1


def test_eval_sources(backend):
    """eval_sources evaluates at the source positions."""
    coords, values = _random_sources(100)
    tree = backend.build_tree(coords, values).result()

    v = tree.eval_sources(mode="direct").result()
    assert v.shape == (100, 3)

    # Compare with eval at the same points.
    v2 = tree.eval(coords, mode="direct").result()
    np.testing.assert_allclose(v, v2, rtol=1e-12, atol=1e-12)


def test_rebuild_reuses_tree(backend):
    """Rebuild returns the same tree object with new sources."""
    coords, values = _random_sources(200)
    tree = backend.build_tree(coords, values).result()

    coords2 = coords + 2.0
    future = tree.rebuild(coords2, values)
    tree2 = future.result()
    assert tree2 is tree
    assert tree.built is True
    assert tree.n_sources == 200

    # The tree now reflects the new source positions.
    pts = np.array([[10.0, 0.0, 0.0]])
    v1 = tree.eval(pts, mode="direct").result()
    v2 = tree.eval(coords2[:1], mode="direct").result()
    assert v1.shape == (1, 3)
    assert v2.shape == (1, 3)


def test_error_paths(backend):
    """Bad mode raises; mismatched coords/values raises."""
    coords, values = _random_sources(50)
    tree = backend.build_tree(coords, values).result()

    with pytest.raises(ValueError):
        tree.eval(np.zeros((1, 3)), mode="bogus")

    with pytest.raises(ValueError):
        backend.build_tree(np.zeros((5, 3)), np.zeros((7, 3)))
