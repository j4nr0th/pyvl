.. _pyvl.public_python:

.. currentmodule:: pyvl


Public Python API
=================

This part of the documentation describes sections of the API which are
intended for a typical user.

Positioning Geometry
--------------------

Before there can be any talk of defining geometry, it must be first positioned in space.
If geometry consists of several components, it might be useful to define those in their
own coordinate system. Even if only a single component is analyzed, it might be defined
in a different coordinate system than expected. Take for example the absolute lunatics
known as flight dynamics engineers. These lot would have you believe that the z-axis
for a plane ought to be pointing towards the ground, while the x-axis is positive from
the tail towards the plane's nose.

To be able to co-exist with such savages, the :class:`ReferenceFrame` object can be used.
The :class:`ReferenceFrame` can also be subclassed to allow for translational and rotational
motion. This is described more in depth on :ref:`this page <pyvl.reference_frame>`.


Defining Geometry
-----------------

The most basic building block is the :class:`Geometry` object. Since
``PyVL`` is not a meshing library, the expectation is that the mesh will
be created with some other program, then loaded by either using
:class:`pyvista.PolyData` or :class:`meshio.Mesh`. More information about the
:class:`Geometry` type can be found :ref:`here <pyvl.geometry>`.

One or more :class:`Geometry` objects constitute a :class:`SimulationGeometry`. This
is an object, which can be passed to the solver and computes additional information about
the individual geometrical objects, such as the dual mesh.

Solver Configuration
--------------------

To configure the non-geometry related solver settings, the :class:`SolverSettings` object
can be used. This object contains other settings-related objects, which contain settings
related to different aspects of the solver.

While the other categories are described below, more information about :class:`SolverSettings`
can be found :ref:`here <pyvl.solver_settings>`

Describing Flow
~~~~~~~~~~~~~~~

Besides geometry, the flow conditions need to be described. This is done by by using a sub-type
of :class:`FlowConditions`. This is an abstract base class (inherits from :class:`abc.ABC`), which
means it defines methods which must be implemented. For most basic cases this is un-necessary, as
almost all common cases can be easily be represented by the sub-types implemented in the :mod:`pyvl`
module. More information about the base class and sub-types already implemented is provided
:ref:`here <pyvl.flow_conditions>`.

Simulation Times
~~~~~~~~~~~~~~~~

To set the duration of the simulation, the :class:`TimeSettings` object is used. It specifies
the number of time incriments as well as their number. It can also be optionally used to specify
how often output of the simulation is saved. If it is not specified, each step will be saved,
otherwise, it will only be saved one every ``n`` step.

It can also be used to run multiple steady-state simulations in series, depending on the choice
of :ref:`flow conditions <pyvl.flow_conditions>` and :ref:`wake model <pyvl.wake_models>`. More details about the class
can be found :ref:`here <pyvl.time_settings>`.

Specifying Models
~~~~~~~~~~~~~~~~~

The potential flow itself is a model of a flow. As such, there are some settings which can be
changed. These hyper-parameters are all gathered in the :class:`ModelSettings` objects. The most
important one is the :attr:`ModelSettings.vortex_limit`, which is the distance at which the
line vortex velocity is set to zero instead of its usual :math:`\frac{1}{r}` scaling.

More details about these can be read about :ref:`here <pyvl.model_settings>`

Defining the Wake
~~~~~~~~~~~~~~~~~

One of the most important things to model is the wake. The approach that `pyvl` has to modeling
wake is that any object which inherits from :class:`WakeModel` can be used. As such, it is possible
to define a custom wake model if that is required, however, that might not be needed at all for most
cases, as the most common wake models are already implemented. For a more in-depth details about the
different wake models available, see :ref:`this <pyvl.wake_models>`.

Controlling the Output
----------------------

While controlling the frequency of output is up to :class:`TimeSettings`, the manner in which it is done
is controlled by :class:`OutputSettings`. Simulation results can theoretically be written to any file type
which supports hierarchical data storage along with number arrays.

Currently, IO can be done with the `JSON <https://www.json.org/>`_ and `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
file formats. More specific information about these files can be found :ref:`here <pyvl.output_settings>`

Running the Solver
------------------

After the geometry is defined and all the settings have been assembled, the simulation can be run
using the :func:`run_solver`. More information about it and using the solver is provided
:ref:`here <pyvl.solver>`.

.. toctree::
    :maxdepth: 2
    :hidden:

    reference_frame
    geometry
    flow_conditions
    time_settings
    model_settings
    solver_settings
    wake_models
    output_settings
    solver
