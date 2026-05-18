"""Tests related to the ReferenceFrame class."""

import numpy as np
import pytest
from pyvl import ReferenceFrame


def test_creation_and_getset():
    """Test that it can be created and that getters/setters work as expected."""
    theta_x = 0.2
    theta_y = 0.3
    theta_z = -2
    offset_x = 3
    offset_y = 2
    offset_z = 0
    rf_0 = ReferenceFrame(
        offset=(offset_x, offset_y, offset_z),
        theta=(theta_x, theta_y, theta_z),
    )
    assert all(rf_0.offset_at() == (offset_x, offset_y, offset_z))

    # We do angles this way because we can clip them to [0, 2pi] range.
    angles = rf_0.angles_at()
    assert np.sin(angles) == pytest.approx(np.sin([theta_x, theta_y, theta_z]))
    assert np.cos(angles) == pytest.approx(np.cos([theta_x, theta_y, theta_z]))


def test_parents():
    """Test that parent-related functions work."""
    rng = np.random.default_rng(0)
    rf_0 = ReferenceFrame(rng.random(3), rng.random(3), rng.random(3), rng.random(3))
    rf_1 = ReferenceFrame((102, 4.20, 1.4), (31.2, 33.0, -2), parent=rf_0)
    rf_2 = ReferenceFrame((-102, -4.20, 1.4), (-31.2, 33.0, 2), parent=rf_1)

    assert rf_2.parent is rf_1
    assert rf_2.parents == (rf_1, rf_0)


def test_rotate_by_and_offset():
    """Test that rotation and offset changes work."""
    rng = np.random.default_rng(14)
    rf_0 = ReferenceFrame(rng.random(3), rng.random(3))
    rf_1 = ReferenceFrame((31.2, 33.0, -2), (0, 0, 0), parent=rf_0)
    rf_2 = rf_1.rotate_x(2.1)
    rf_3 = rf_1.rotate_y(1.1)
    rf_4 = rf_1.rotate_z(0.1)
    rf_5 = rf_1.with_offset([0, 2, 3.0])

    angles_1 = rf_1.angles_at()
    angles_2 = rf_2.angles_at()
    offset_1 = rf_1.offset_at()
    offset_2 = rf_2.offset_at()

    assert rf_2.parent is rf_1.parent
    assert all(offset_2 == offset_1)
    assert angles_2[1] == pytest.approx(angles_1[1])
    assert angles_2[2] == pytest.approx(angles_1[2])
    assert angles_2[0] == pytest.approx(angles_1[0] + 2.1)

    angles_3 = rf_3.angles_at()
    assert rf_3.parent is rf_1.parent
    assert all(rf_3.offset_at() == offset_1)
    assert angles_3[0] == pytest.approx(angles_1[0])
    assert angles_3[1] == pytest.approx(angles_1[1] + 1.1)
    assert angles_3[2] == pytest.approx(angles_1[2])

    angles_4 = rf_4.angles_at()
    assert rf_4.parent is rf_1.parent
    assert all(rf_4.offset_at() == offset_1)
    assert angles_4[0] == pytest.approx(angles_1[0])
    assert angles_4[1] == pytest.approx(angles_1[1])
    assert angles_4[2] == pytest.approx(angles_1[2] + 0.1)

    assert rf_5.parent is rf_1.parent
    assert all(rf_5.offset_at() == [0, 2, 3.0])
    assert all(rf_5.angles_at() == angles_1)


def test_rotation_is_orthonormal():
    """Check that for all angles the rotation is orthonormal."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        rf = ReferenceFrame(theta=rng.random(3))
        rot_mat = rf.rotation_matrix_at()
        # Must be orthonormal
        assert pytest.approx(rot_mat @ rot_mat.T) == np.eye(3)


def test_transformations_are_inverse():
    """Check that transformations of reference frames are inverse."""
    rng = np.random.default_rng(592)
    for _ in range(10):
        rf = ReferenceFrame(
            rng.random(3),
            rng.random(3),
            rng.random(3),
            rng.random(3),
        )
        x = rng.random((12, 51, 2, 3))
        v = rng.random((12, 51, 2, 3))
        x1, v1 = rf.to_parent_velocity(*rf.from_parent_velocity(x, v))
        assert x1 == pytest.approx(x)
        assert v1 == pytest.approx(v)
        assert pytest.approx(x) == rf.to_parent_position(rf.from_parent_position(x))
        assert pytest.approx(x) == rf.to_parent_vector(rf.from_parent_vector(x))


def test_transformation_output():
    """Check that transformation function with out argument behave properly."""
    rng = np.random.default_rng(124590)
    rf = ReferenceFrame(rng.random(3), rng.random(3))
    real_shape_in = (3, 1, 4, 10, 3)
    x_in = rng.random(real_shape_in)
    # Passing some random object won't work as second positional arg (time)
    caught = False
    try:
        _ = rf.to_parent_position(x_in, "SOME random object")
    except TypeError:
        caught = True
    assert caught

    caught = False

    # Passing array-like also won't work
    caught = False
    try:
        _ = rf.to_parent_position(x_in, out=[0, 1, [0, 2, 3]])
    except TypeError:
        caught = True
    assert caught

    # Passing array of wrong shape as out
    caught = False
    try:
        _ = rf.to_parent_position(x_in, out=np.array([[0, 2, 3]]))
    except ValueError:
        caught = True
    assert caught

    # Passing array of wrong data type
    caught = False
    try:
        x_out = np.empty_like(x_in, dtype=np.float32)
        _ = rf.to_parent_position(x_in, out=x_out)
    except ValueError:
        caught = True
    assert caught

    # Passing array that is non-contiguous won't work either
    caught = False
    try:
        x_out = np.empty(real_shape_in + (4,), dtype=np.float64)
        _ = rf.to_parent_position(x_in, out=(x_out[..., 2]).reshape(real_shape_in))
    except ValueError:
        caught = True
    assert caught

    # Reference to the array should still be returned
    x_out = np.empty_like(x_in)
    res_out = rf.to_parent_position(x_in, out=x_out)
    assert res_out is x_out

    # Result must be identical when no output array is given
    assert np.all(res_out == rf.to_parent_position(x_in))

    # Must be possible to call inplace
    rf.to_parent_position(x_in, out=x_in)
    assert np.all(res_out == x_in)


def test_simple_transformations():
    """Manually check some basic transformations."""
    eye = np.eye(3)
    rf1 = ReferenceFrame(offset=(0, 1.0, 0), theta=(np.pi / 2, 0, 0))
    eye2 = rf1.from_parent_position(eye)
    assert pytest.approx(eye2) == [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    eye2 = rf1.from_parent_vector(eye)
    assert pytest.approx(eye2) == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    rf1 = ReferenceFrame(theta=(0.0, 0, np.pi / 2))
    out = np.empty_like(eye)
    eye2 = rf1.from_parent_position(eye, out=out)
    assert pytest.approx(eye2) == [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    eye2 = rf1.from_parent_vector(eye, out=out)
    assert pytest.approx(eye2) == [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def test_angles_from_rotation():
    """Check that the static method angles_from_rotation recovers the angles."""
    rng = np.random.default_rng(0)
    rf = ReferenceFrame(theta=rng.random(3))

    rot_mat = rf.rotation_matrix_at()
    recovered = ReferenceFrame.angles_from_rotation(rot_mat)
    assert rf.angles_at() == pytest.approx(recovered)


def test_time_varying_reference_frame():
    """Test callable-based reference frame."""

    def pos_func(t):
        return (t, 2 * t, 3 * t)

    def ori_func(t):
        return (t * 0.1, t * 0.2, t * 0.3)

    rf = ReferenceFrame(offset=pos_func, theta=ori_func)

    # Check at t=0
    pos0 = rf.offset_at(0.0)
    ori0 = rf.angles_at(0.0)
    assert pos0 == pytest.approx([0, 0, 0])
    assert ori0 == pytest.approx([0, 0, 0])

    # Check at t=1
    pos1 = rf.offset_at(1.0)
    ori1 = rf.angles_at(1.0)
    assert pos1 == pytest.approx([1, 2, 3])
    assert ori1 == pytest.approx([0.1, 0.2, 0.3])

    # Check at t=2
    pos2 = rf.offset_at(2.0)
    assert pos2 == pytest.approx([2, 4, 6])


def test_time_parameter_in_transformations():
    """Test that transformation methods accept time parameter."""

    def pos_func(t):
        return (t, 0, 0)

    rf = ReferenceFrame(offset=pos_func, theta=(0, 0, 0))

    x = np.array([1.0, 2.0, 3.0])

    # At t=0, offset is 0, so from_parent_with_offset(x, time=0) = x
    result_t0 = rf.from_parent_position(x, time=0.0)
    assert result_t0 == pytest.approx([1, 2, 3])

    # At t=1, offset is (1, 0, 0), so from_parent_with_offset(x, time=1) = x + (1, 0, 0)
    result_t1 = rf.from_parent_position(x, time=1.0)
    assert result_t1 == pytest.approx([2, 2, 3])
