"""Tests related to wake models."""

import numpy as np

# import pyvista as pv
from pyvl import (
    Geometry,
    Mesh,
    ModelSettings,
    ReferenceFrame,
    SimulationGeometry,
    SolverSettings,
    WakeModelLineExplicitUnsteady,
)
from pyvl.flow_conditions import FlowConditionsUniform


def test_explicit_unsteady():
    """Test the WakeModelLineExplicitSteady behaviour."""
    node_positions = np.array(
        [
            [0, 1, 0],
            [3, 1, -0.5],
            [3, 1, +0.5],
            [0, -1, 0],
            [3, -1, -0.5],
            [3, -1, +0.5],
        ]
    )
    geo = Geometry(
        "wedge",
        ReferenceFrame(),
        Mesh(6, [[0, 3, 5, 2], [3, 0, 1, 4], [1, 2, 5, 4]]),
        node_positions,
    )
    sim_geo = SimulationGeometry(geo)

    trailing_edge = sim_geo.dual.dual_normal_criterion(
        -0.5, sim_geo.mesh.surface_normal(sim_geo.positions_at_time(0))
    )
    # print(
    #     f"Trailing edge has id(s) {trailing_edge}, which is the line"
    #     f" {sim_geo.mesh.get_line(trailing_edge[0])}"
    # )

    assert len(trailing_edge) == 1
    ln = sim_geo.mesh.get_line(trailing_edge[0])
    assert (ln.begin == 0 and ln.end == 3) or (ln.begin == 3 and ln.begin == 0)
    fc = FlowConditionsUniform(-1, 0, 0)
    settings = SolverSettings(fc, ModelSettings(1e-6))
    shedding_lines = sim_geo.te_normal_criterion(-0.5)
    nds, srf = sim_geo.line_adjecency_information(shedding_lines)
    wake_model = WakeModelLineExplicitUnsteady(
        nds, shedding_lines, srf, settings.model_settings.vortex_limit, -1, 5
    )

    def _circulation_function(time: float):
        """Return changing circulation."""
        return 0.01 * np.array((np.sin(time), np.cos(time), 1))

    for i in range(5):
        wake_model.update(
            float(i), sim_geo, sim_geo.positions_at_time(0), _circulation_function(i), fc
        )

        # plt = pv.Plotter()
        # pd = geo.as_polydata()
        # plt.add_mesh(pd, color="red", label="SimGeo")
        # wake_mesh = wake_model.as_polydata()
        # wake_mesh.set_active_scalars("circulation")
        # plt.add_mesh(wake_mesh, label="WakeModel")
        # plt.show()

    # print("Checking the difference of circulations.")
    for i in range(5):
        v1, v2, _ = _circulation_function(i)
        assert wake_model.circulation[0, 4 - i] == (v1 - v2)
    # print("Circulation is being properly differenced.")


def test_explicit_unsteady2():
    """Test the WakeModelLineExplicitSteady behaviour."""
    node_positions = np.array(
        [
            [0, 1, 0],  # 0
            [3, 1, -0.5],  # 1
            [3, 1, +0.5],  # 2
            [0, -1, 0],  # 3
            [3, -1, -0.5],  # 4
            [3, -1, +0.5],  # 5
            [0, -2, 0],  # 6
            [3, -2, -0.5],  # 7
            [3, -2, +0.5],  # 8
        ]
    )
    geo = Geometry(
        "wedge",
        ReferenceFrame(),
        Mesh(
            9,
            [
                [0, 3, 5, 2],
                [3, 0, 1, 4],
                [1, 2, 5, 4],
                [3, 6, 8, 5],
                [6, 3, 4, 7],
                [4, 5, 8, 7],
            ],
        ),
        node_positions,
    )
    sim_geo = SimulationGeometry(geo)

    trailing_edge = sim_geo.dual.dual_normal_criterion(
        -0.5, sim_geo.mesh.surface_normal(sim_geo.positions_at_time(0))
    )
    # print(
    #     f"Trailing edge has id(s) {trailing_edge}, which is the line"
    #     f" {sim_geo.mesh.get_line(trailing_edge[0])}"
    # )

    assert len(trailing_edge) == 2
    ln = sim_geo.mesh.get_line(trailing_edge[0])
    assert (ln.begin == 0 and ln.end == 3) or (ln.begin == 3 and ln.begin == 0)
    ln = sim_geo.mesh.get_line(trailing_edge[1])
    assert (ln.begin == 3 and ln.end == 6) or (ln.begin == 6 and ln.begin == 3)
    fc = FlowConditionsUniform(-1, 0, 0)
    settings = SolverSettings(fc, ModelSettings(1e-6))
    shedding_lines = sim_geo.te_normal_criterion(-0.5)
    nds, srf = sim_geo.line_adjecency_information(shedding_lines)
    wake_model = WakeModelLineExplicitUnsteady(
        nds, shedding_lines, srf, settings.model_settings.vortex_limit, -1, 5
    )

    def _circulation_function(time: float):
        """Return changing circulation."""
        return 0.01 * np.array(
            (np.sin(time), np.cos(time), 1, np.cos(time), np.sin(time), -1)
        )

    for i in range(5):
        wake_model.update(
            float(i), sim_geo, sim_geo.positions_at_time(0), _circulation_function(i), fc
        )

        # plt = pv.Plotter()
        # pd = geo.as_polydata()
        # plt.add_mesh(pd, color="red", label="SimGeo")
        # wake_mesh = wake_model.as_polydata()
        # wake_mesh.set_active_scalars("circulation")
        # plt.add_mesh(wake_mesh, label="WakeModel")
        # plt.show()

    # print("Checking the difference of circulations.")
    for i in range(5):
        v1, v2, _, v3, v4, _ = _circulation_function(i)
        assert all(wake_model.circulation[:, 4 - i] == ((v1 - v2, v3 - v4)))
    # print("Circulation is being properly differenced.")
