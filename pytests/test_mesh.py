"""Tests related to the Mesh object."""

import numpy as np
from pydust import Mesh


def test_mesh_construction():
    """Check that constructor and getters/setters works as intended."""
    positions = np.array(
        [[0, 2, 0.5], [0.4, 2, 0.4], [4, 0.1, 0.2], [3, -0.1, 0.2], [2, 1, 0.2]]
    )
    elements = [[0, 1, 2, 3], [2, 3, 4], [0, 1, 4], [2, 4, 5]]
    msh = Mesh(positions, elements)
    assert np.all(msh.positions == positions)
    element_count, _ = msh.to_element_connectivity()
    assert np.all(element_count == [len(e) for e in elements])
