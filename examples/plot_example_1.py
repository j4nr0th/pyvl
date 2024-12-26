r""".. pyvl.examples.1:.

Flat Plate
==========

The first case which is typically analyzed in aerodynamics is the simple flat plate.
This can also serve as validation for any solver, as for incompressible, non-viscous,
steady flow, the lift coefficient of a infinitely thin flat plate should be based solely
on the inflow angle of attack :math:`\alpha` :

.. math::

    C_l = 2 \pi \sin{\alpha}

.. currentmodule:: pyvl
"""

import numpy as np
import pyvista as pv
import pyvl

pv.set_plot_theme("document")
pv.set_jupyter_backend("html")

# %%
#
# Geometry Setup
# --------------
#
# The first step is to set up the :class:`Geometry` of the simulation. PyVL is not
# intended to be a mesh generator. As such, the geometry can be loaded either using
# :mod:`pyvista` or `MeshIO <https://pypi.org/project/meshio/>` modules.
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
# First, there's the :class:`FlowConditions`. This is an :class:`ABC` intended to be
# subclassed in case anything more specific is required. If a constant free-stream
# velocity is good enough, the module provides :class:`FlowConditionsUniform`, which can
# be used for constant free-stream.

alpha = np.radians(5)  # 5 degrees
v_inf = 10  # 10 m/s
flow_conditions = pyvl.FlowConditionsUniform(
    v_inf * np.cos(alpha), v_inf * np.sin(alpha), 0
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
# Besides the geometry and solver related settings, the last thing to specify are
# the settings related to solver outputs. As the solver runs, it will output its
# state at interval specified by :class:`TimeSettings`. In this case, it was not
# specified, which means that each time step will be saved.
#
# The output settings are specified by :class:`OutputSettings`. The object is instantiated
# by specifying two things:
#
# - What file format to output the data in ("JSON" or "HDF5"),
# - How to name files based on the iteration number and time (via a callback).

output_settings = pyvl.OutputSettings("JSON", lambda i, _: f"/tmp/output-{i:d}.json")

# %%
#
# Running the Solver
# ------------------
#
# The solver can now be run by calling :func:`run_solver` and passing the
# :class:`SimulationGeometry`, :class:`SolverSettings`, and :class:`OutputSettings`.


pyvl.run_solver(sim_geo, settings, output_settings)
