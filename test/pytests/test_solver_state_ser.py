"""Test the serialization and deserialization of SolverState."""

import numpy as np
import pytest
from pyvl import Geometry, ReferenceFrame
from pyvl.cvl import TransformationPlane
from pyvl.fio.io_common import PythonSerializer
from pyvl.geometry import SimulationGeometry
from pyvl.settings import ModelSettings, SolverSettings, WakeSettings, WakeShedderUniform
from pyvl.solver import SolverState


def create_simple_geometry() -> SimulationGeometry:
    """Create a simple plane."""
    pos = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.double,
    )

    # Mesh connectivity for a single quad (two triangles)
    # In pyvl, Mesh is typically created from connectivity lists.
    # Using a simple 2-triangle quad.
    from pyvl.cvl import Mesh

    connectivity = [
        np.array([0, 1, 2], dtype=np.uint32),
        np.array([0, 2, 3], dtype=np.uint32),
    ]
    msh = Mesh(n_points=4, connectivity=connectivity)

    geo = Geometry(
        "plane",
        ReferenceFrame(),
        msh,
        pos,
    )
    return SimulationGeometry.from_geometries(geo)


def test_solver_state_serialization():
    """Check the entire state is serialized and deserialized correctly."""
    sim_geo = create_simple_geometry()
    s_settings = SolverSettings(
        flow_velocity=(10.0, 0, 0),
        model_settings=ModelSettings(
            vortex_cutoff=1e-6,
            vortex_far_approximation=1e-6,
            vortex_smallest_size=1e-6,
            symmetry_plane=TransformationPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        wake_settings=WakeSettings(WakeShedderUniform([3, 2, 1]), 31),
    )
    t = 3.21

    state = SolverState.create_new(t, sim_geo, s_settings)
    state.circulation[:] = np.random.random(sim_geo.n_lines)

    serializer = PythonSerializer()
    hmap = state.save(serializer.serialize)

    state_in = SolverState.load(hmap, serializer.deserialize)

    assert pytest.approx(state_in.circulation) == state.circulation
    assert state_in.geometry == sim_geo
    assert pytest.approx(state_in.time) == t
    assert state.settings.model_settings.symmetry_plane is not None
    assert state_in.settings.model_settings.symmetry_plane is None


if __name__ == "__main__":
    test_solver_state_serialization()
