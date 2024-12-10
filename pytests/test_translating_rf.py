"""Test more reference frame classes."""

import numpy as np
from pydust.reference_frames import TranslatingReferenceFrame


def test_motion_1():
    """Test moving reference frame with motion only."""
    np.random.seed(128094)
    offsets = np.random.random_sample(3)
    angles = np.random.random_sample(3)
    velocity = np.random.random_sample(3)
    times = np.random.random_sample(2)
    rf = TranslatingReferenceFrame(
        offset=offsets,
        theta=angles,
        time=times[0],
        velocity=velocity,
    )
    new = rf.at_time(times[1])
    assert all(new.offset == offsets + velocity * (times[1] - times[0]))
    assert all(new.angles == angles)
    assert new.time == times[1]
