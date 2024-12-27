"""Test more reference frame classes."""

import numpy as np
import pytest
from pyvl.reference_frames import RotorReferenceFrame
from scipy.integrate import quad


def test_rotor_1():
    """Test rotor reference frame with constant rate of rotation."""
    np.random.seed(128094)
    offsets = np.random.random_sample(3)
    angles = np.random.random_sample(3)
    omega = np.random.random_sample(1)[0]
    times = np.random.random_sample(2)
    rf = RotorReferenceFrame(
        rotation=lambda t: omega,  # noqa: ARG005
        offset=offsets,
        theta=angles,
        time=times[0],
    )
    new = rf.at_time(times[1])
    assert all(new.offset == offsets)
    a = np.array(angles)
    a[2] += omega * (times[1] - times[0])
    assert new.angles == pytest.approx(a)
    assert new.time == times[1]


def test_rotor_2():
    """Test rotor reference frame with no rotation."""
    np.random.seed(128094)
    offsets = np.random.random_sample(3)
    angles = np.random.random_sample(3)
    times = np.random.random_sample(2)
    rf = RotorReferenceFrame(
        rotation=None,
        offset=offsets,
        theta=angles,
        time=times[0],
    )
    new = rf.at_time(times[1])
    assert all(new.offset == offsets)
    assert all(new.angles == angles)
    assert new.time == times[1]


def test_rotor_3():
    """Test rotor reference frame with horrible rotation."""
    np.random.seed(128094)

    def rotate_function(t):
        """Return non-constant rotation velocity."""
        return 3 * np.sin(np.pi * t**2) - t / (3 + t**2)

    offsets = np.random.random_sample(3)
    angles = np.random.random_sample(3)
    times = np.random.random_sample(2)
    rf = RotorReferenceFrame(
        rotation=rotate_function,
        offset=offsets,
        theta=angles,
        time=times[0],
    )
    new = rf.at_time(times[1])
    assert all(new.offset == offsets)
    a = np.array(angles)
    a[2] += quad(rotate_function, times[0], times[1])[0]
    assert new.angles == pytest.approx(a)
    assert new.time == times[1]
