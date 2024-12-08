"""Tests related to the ReferenceFrame class."""

import numpy as np
import pytest
from pydust import ReferenceFrame


def test_creation_and_getset():
    """Test that it can be created and that getters/setters work as expected."""
    theta_x = 0.2
    theta_y = 0.3
    theta_z = -2
    offset_x = 3
    offset_y = 2
    offset_z = 0
    rf_0 = ReferenceFrame(
        theta_z=theta_z,
        offset_z=offset_z,
        offset_x=offset_x,
        theta_y=theta_y,
        theta_x=theta_x,
        offset_y=offset_y,
    )
    # Test sines and cosines, since negative angles will be wrapped to the positive side.
    assert np.sin(rf_0.angles) == pytest.approx(np.sin([theta_x, theta_y, theta_z]))
    assert np.cos(rf_0.angles) == pytest.approx(np.cos([theta_x, theta_y, theta_z]))
    assert all(rf_0.offset == (offset_x, offset_y, offset_z))


def test_parents():
    """Test that parent-related functions work."""
    np.random.seed(0)
    rf_0 = ReferenceFrame(*np.random.random_sample(6))
    rf_1 = ReferenceFrame(102, 4.20, 1.4, 31.2, 33.0, -2, rf_0)
    rf_2 = ReferenceFrame(-102, -4.20, 1.4, -31.2, 33.0, 2, rf_1)

    assert rf_2.parent is rf_1
    assert rf_2.parents == (rf_1, rf_0)


def test_rotate_by_and_offset():
    """Test that rotation and offset changes work."""
    rf_0 = ReferenceFrame(*np.random.random_sample(6))
    rf_1 = ReferenceFrame(0, 0, 0, 31.2, 33.0, -2, rf_0)
    rf_2 = rf_1.rotate_x(2.1)
    rf_3 = rf_1.rotate_y(1.1)
    rf_4 = rf_1.rotate_z(0.1)
    rf_5 = rf_1.with_offset([0, 2, 3.0])

    assert rf_2.parent is rf_1.parent
    assert all(rf_2.offset == rf_1.offset)
    assert all(rf_2.angles[1:] == rf_1.angles[1:])
    assert rf_2.angles[0] == rf_1.angles[0] + 2.1

    assert rf_3.parent is rf_1.parent
    assert all(rf_3.offset == rf_1.offset)
    assert rf_3.angles[0] == rf_1.angles[0]
    assert rf_3.angles[1] == rf_1.angles[1] + 1.1
    assert rf_3.angles[2] == rf_1.angles[2]

    assert rf_4.parent is rf_1.parent
    assert all(rf_4.offset == rf_1.offset)
    assert rf_4.angles[0] == rf_1.angles[0]
    assert rf_4.angles[1] == rf_1.angles[1]
    assert rf_4.angles[2] == rf_1.angles[2] + 0.1

    assert rf_5.parent is rf_1.parent
    assert all(rf_5.offset == [0, 2, 3.0])
    assert all(rf_5.angles == rf_1.angles)


def test_rotation_is_orthonormal():
    """Check that for all angles the rotation is orthonormal."""
    np.random.seed(0)
    for _ in range(100):
        rf = ReferenceFrame(*np.random.random_sample(3))
        rot_mat = rf.rotation_matrix
        # Must be orthonormal
        assert pytest.approx(rot_mat @ rot_mat.T) == np.eye(3)


def test_transformations_are_inverse():
    """Check that transformations of reference frames are inverse."""
    np.random.seed(592)
    for _ in range(10):
        rf = ReferenceFrame(*np.random.random_sample(6))
        x = np.random.random_sample((12, 51, 2, 3))
        assert pytest.approx(x) == rf.to_parent_with_offset(rf.from_parent_with_offset(x))
        assert pytest.approx(x) == rf.to_parent_without_offset(
            rf.from_parent_without_offset(x)
        )


def test_transformation_output():
    """Check that transformation function with out argument behave properly."""
    np.random.seed(124590)
    rf = ReferenceFrame(*np.random.random_sample(6))
    real_shape_in = (3, 1, 4, 10, 3)
    x_in = np.random.random_sample(real_shape_in)
    # Passing some random object won't work
    caught = False
    try:
        _ = rf.to_parent_with_offset(x_in, "SOME random object")
    except TypeError:
        caught = True
    assert caught

    caught = False

    # Passing array-like also won't work
    caught = False
    try:
        _ = rf.to_parent_with_offset(x_in, [0, 1, [0, 2, 3]])
    except TypeError:
        caught = True
    assert caught

    # Passing array of wrong shape won't work
    caught = False
    try:
        _ = rf.to_parent_with_offset(x_in, np.array([[0, 2, 3]]))
    except ValueError:
        caught = True
    assert caught

    # Passing array of wrong data type
    caught = False
    try:
        x_out = np.empty_like(x_in, dtype=np.float32)
        _ = rf.to_parent_with_offset(x_in, x_out)
    except ValueError:
        caught = True
    assert caught

    # Passing array that is non-contiguous won't work either
    caught = False
    try:
        x_out = np.empty(real_shape_in + (4,), dtype=np.float64)
        _ = rf.to_parent_with_offset(x_in, (x_out[..., 2]).reshape(real_shape_in))
    except ValueError:
        caught = True
    assert caught

    # Reference to the array should still be returned
    x_out = np.empty_like(x_in)
    res_out = rf.to_parent_with_offset(x_in, x_out)
    assert res_out is x_out

    # Result must be identical when no output array is given
    assert np.all(res_out == rf.to_parent_with_offset(x_in))

    # Must be possible to call inplace
    rf.to_parent_with_offset(x_in, x_in)
    assert np.all(res_out == x_in)


def test_simple_transformations():
    """Manually check some basic transformations."""
    eye = np.eye(3)
    rf1 = ReferenceFrame(np.pi / 2, offset_y=1.0)
    eye2 = rf1.from_parent_with_offset(eye)
    assert pytest.approx(eye2) == [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    eye2 = rf1.from_parent_without_offset(eye)
    assert pytest.approx(eye2) == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    rf1 = ReferenceFrame(theta_z=np.pi / 2)
    eye2 = rf1.from_parent_with_offset(eye, eye2)
    assert pytest.approx(eye2) == [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    eye2 = rf1.from_parent_without_offset(eye, eye2)
    assert pytest.approx(eye2) == [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
