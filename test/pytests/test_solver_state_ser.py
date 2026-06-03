"""Test the serialization and deserialization of SolverState and ImplicitElements."""

import numpy as np
from pyvl import Geometry, ReferenceFrame, flow_conditions, settings
from pyvl.elements import ImplicitElements
from pyvl.fio.io_common import PythonSerializer
from pyvl.geometry import SimulationGeometry
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
        dtype=np.float64,
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
    return SimulationGeometry(geo)


def test_implicit_elements_serialization():
    """Check elements are serialized and deserialized correctly."""
    sim_geo = create_simple_geometry()
    elements = ImplicitElements.from_geometry(sim_geo)

    serializer = PythonSerializer()
    # Save
    hmap = elements.save(serializer.serialize)

    # Load
    elements_in = ImplicitElements.load(hmap, serializer.deserialize)

    assert np.allclose(elements.positions, elements_in.positions)
    assert np.allclose(elements.normals, elements_in.normals)
    assert np.allclose(elements.control_points, elements_in.control_points)


def test_solver_state_serialization():
    """Check the entire state is serialized and deserialized correctly."""
    sim_geo = create_simple_geometry()
    s_settings = settings.SolverSettings(
        flow_conditions=flow_conditions.FlowConditionsUniform(10.0, 0, 0),
        model_settings=settings.ModelSettings(vortex_limit=1e-6),
        time_settings=settings.TimeSettings(nt=10, dt=0.1),
    )

    state = SolverState(sim_geo, s_settings, None)
    state.iteration = 5
    state.circulation[:] = np.random.random(sim_geo.n_surfaces)
    state.cp_velocity[:] = np.random.random((sim_geo.n_surfaces, 3))

    # Simulate movement to set current_elements
    t = 0.5
    state.current_elements, state.cp_velocity = state.current_elements.at_time(
        t, out_v=state.cp_velocity
    )

    serializer = PythonSerializer()
    hmap = state.save(serializer.serialize)

    state_in = SolverState.load(hmap, serializer.deserialize)

    assert state_in.iteration == state.iteration
    assert np.allclose(state_in.circulation, state.circulation)
    assert np.allclose(state_in.cp_velocity, state.cp_velocity)
    assert np.allclose(
        state_in.current_elements.positions, state.current_elements.positions
    )
    assert np.allclose(state_in.current_elements.normals, state.current_elements.normals)
    assert np.allclose(
        state_in.current_elements.control_points, state.current_elements.control_points
    )
    assert state_in.geometry == sim_geo
