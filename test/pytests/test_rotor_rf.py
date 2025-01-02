"""Test more reference frame classes."""

import numpy as np
import pytest
from pyvl.reference_frames import RotorReferenceFrame


def test_internal_rotation_z():
    """Check that internal rotation function works around z."""
    omega = np.array((0, 0, 1), np.float64)

    assert RotorReferenceFrame._rotate(
        omega, np.array((1.0, 0.0, 0.0)), 0.0
    ) == pytest.approx((1, 0, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((1.0, 0.0, 0.0)), np.pi / 2
    ) == pytest.approx((0, 1, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((1.0, 0.0, 0.0)), np.pi
    ) == pytest.approx((-1, 0, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((1.0, 0.0, 0.0)), 2 * np.pi
    ) == pytest.approx((1, 0, 0))

    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 1.0, 0.0)), 0.0
    ) == pytest.approx((0, 1, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 1.0, 0.0)), np.pi / 2
    ) == pytest.approx((-1, 0, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 1.0, 0.0)), np.pi
    ) == pytest.approx((0, -1, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 1.0, 0.0)), 2 * np.pi
    ) == pytest.approx((0, 1, 0))


def test_internal_rotation_y():
    """Check that internal rotation function works around y."""
    omega = np.array((0, 1, 0), np.float64)

    assert RotorReferenceFrame._rotate(
        omega, np.array((1.0, 0.0, 0.0)), 0.0
    ) == pytest.approx((1, 0, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((1.0, 0.0, 0.0)), np.pi / 2
    ) == pytest.approx((0, 0, -1))
    assert RotorReferenceFrame._rotate(
        omega, np.array((1.0, 0.0, 0.0)), np.pi
    ) == pytest.approx((-1, 0, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((1.0, 0.0, 0.0)), 2 * np.pi
    ) == pytest.approx((1, 0, 0))

    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 0.0, 1.0)), 0.0
    ) == pytest.approx((0, 0, 1))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 0.0, 1.0)), np.pi / 2
    ) == pytest.approx((1, 0, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 0.0, 1.0)), np.pi
    ) == pytest.approx((0, 0, -1))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 0.0, 1.0)), 2 * np.pi
    ) == pytest.approx((0, 0, 1))


def test_internal_rotation_x():
    """Check that internal rotation function works around x."""
    omega = np.array((1, 0, 0), np.float64)

    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 0.0, 1.0)), 0.0
    ) == pytest.approx((0, 0, 1))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 0.0, 1.0)), np.pi / 2
    ) == pytest.approx((0, -1, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 0.0, 1.0)), np.pi
    ) == pytest.approx((0, 0, -1))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 0.0, 1.0)), 2 * np.pi
    ) == pytest.approx((0, 0, 1))

    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 1.0, 0.0)), 0.0
    ) == pytest.approx((0, 1, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 1.0, 0.0)), np.pi / 2
    ) == pytest.approx((0, 0, 1))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 1.0, 0.0)), np.pi
    ) == pytest.approx((0, -1, 0))
    assert RotorReferenceFrame._rotate(
        omega, np.array((0.0, 1.0, 0.0)), 2 * np.pi
    ) == pytest.approx((0, 1, 0))


def test_rotor_1():
    """Test rotor reference frame with constant rate of rotation."""
    np.random.seed(128094)
    offsets = np.random.random_sample(3)
    angles = np.random.random_sample(3)
    omega = np.random.random_sample(3)
    omega /= np.linalg.norm(omega)
    rf = RotorReferenceFrame(
        omega=omega,
        offset=offsets,
        theta=angles,
        time=0,
    )
    new = rf.at_time(2 * np.pi)
    assert all(new.offset == offsets)
    assert new.angles == pytest.approx(rf.angles)
    assert new.time == 2 * np.pi


def test_rotor_2():
    """Test rotor reference frame with no rotation."""
    np.random.seed(128094)
    offsets = np.random.random_sample(3)
    angles = np.random.random_sample(3)
    times = np.random.random_sample(2)
    rf = RotorReferenceFrame(
        offset=offsets,
        theta=angles,
        time=times[0],
    )
    new = rf.at_time(times[1])
    assert all(new.offset == offsets)
    assert all(new.angles == angles)
    assert new.time == times[1]
