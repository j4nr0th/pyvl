r"""Flat Plate
==========

.. currentmodule:: pyvl

The first case which is typically analyzed in aerodynamics is the simple flat plate.
This can also serve as validation for any solver, as for incompressible, non-viscous,
steady flow, the lift coefficient of a infinitely thin flat plate should be based solely
on the inflow angle of attack :math:`\alpha` :

.. math::

    C_l = 2 \pi \sin{\alpha}
"""  # noqa: D205, D400

import numpy as np
import pyvista as pv
import pyvl

pv.set_plot_theme("document")
pv.set_jupyter_backend("html")
pv.global_theme.show_edges = True

# %%
#
# Geometry Setup
# --------------
#
# The first step is to set up the :class:`Geometry` of the simulation. PyVL is not
# intended to be a mesh generator. As such, the geometry can be loaded either using
# :mod:`pyvista` or `MeshIO <https://pypi.org/project/meshio/>`_ modules.
#
# For this case, :mod:`pyvista` will be used to make a simple flat plate.

plate = pv.Plane()
assert isinstance(plate, pv.PolyData)

geo = pyvl.Geometry.from_polydata(
    label="plate",
    reference_frame=pyvl.ReferenceFrame(),
    pd=plate,
)

plt = pv.Plotter()
plt.add_mesh(geo.as_polydata())
plt.show(interactive=False)
plt.close()

# %%
#
# Next, the created :class:`Geometry` is packed together into :class:`SimulationGeometry`.
# This is done to compute some other properties of the overall geometry behind the scenes,
# but that's not really important as a user.

sim_geo = pyvl.SimulationGeometry(geo)

# %%
#
# Prepare the Settings
# --------------------
#
# After the :class:`SimulationGeometry` is prepared, the simulation settings must be
# configured. This is done via the :class::`SolverSettings` object. This contains other
# sub-objects, which themselves contain settings related to different aspects of the
# solver.
#

# %%
#
# First, there's the :class:`FlowConditions`. This is an :class:`abc.ABC` intended to be
# subclassed in case anything more specific is required. If a constant free-stream
# velocity is good enough, the module provides :class:`FlowConditionsUniform`, which can
# be used for constant free-stream.

alpha = np.radians(15)  # 5 degrees
v_inf = 10  # 10 m/s
flow_conditions = pyvl.FlowConditionsUniform(
    v_inf * np.cos(alpha), 0, v_inf * np.sin(alpha)
)

# %%
#
# Next is the :class:`TimeSettings`. These are not particularly useful for this case,
# since it will just be a steady state simulation, but can be used for unsteady cases,
# or to run different steady state configurations in sequence.

time_settings = pyvl.TimeSettings(1, 1)

# %%
#
# Last which will be discussed here is the :class:`ModelSettings`. This class contains
# settings related to the settings made by the solver when it comes to the models of the
# flow and phyisics.

# Specify the minimum distance before vortex has no more effect.
model_settings = pyvl.ModelSettings(vortex_limit=1e-6)

# %%
#
# These can now be combined togethere into the :class:`SolverSettings` object.

settings = pyvl.SolverSettings(flow_conditions, model_settings, time_settings)

# %%
#
# Running the Solver
# ------------------
#
# The solver can now be run by calling :func:`run_solver` and passing the
# :class:`SimulationGeometry`, :class:`SolverSettings`, and :class:`OutputSettings`.


results = pyvl.run_solver(sim_geo, settings, None, None)

# %%
#
# Post-Processing
# ---------------
#
# Now that the results have been computed, post-processing can be done to obtain some
# more useful results. In this example, this is done by creating a :mod:`pyvista` mesh,
# then computing velocity at each point in the mesh, and plotting it by extracting glyphs
# from it.

mesh = pv.RectilinearGrid(
    np.linspace(-1, 1, 11),
    np.linspace(-1, 1, 11),
    np.linspace(-1, 1, 11),
)

velocities = pyvl.postprocess.compute_velocities(results, mesh.points)

for i in range(velocities.shape[0]):
    plotter = pv.Plotter()

    mesh.point_data["Velocity"] = velocities[i, :, :]
    mesh.set_active_vectors("Velocity")

    sg = sim_geo.polydata_at_time(0.0)

    plotter.add_mesh(mesh.glyph(factor=0.01))
    plotter.add_mesh(sg, label="Geometry", color="Red")

    plotter.show(interactive=False)

# %%
#
# Another quantity of interest is the force distribution over the
# mesh. This can be extracted by using :func:`pyvl.postprocess.circulatory_forces`.

forces = pyvl.postprocess.circulatory_forces(results)

for field in forces:
    sg = sim_geo.polydata_edges_at_time(0.0)
    sg.cell_data["Forces"] = field
    print(f"Total force: {np.sum(field, axis=0)} Newtons")
    plotter = pv.Plotter()

    plotter.add_mesh(sg.glyph(factor=1))
    plotter.add_mesh(sg, label="Geometry", color="Red")

    plotter.show(interactive=False)
