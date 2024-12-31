"""Submodule with implementation of post processing analysis."""

# Force Field
from pyvl.postprocess.forces import circulatory_forces as circulatory_forces

# Pressure
from pyvl.postprocess.pressure import (
    compute_dynamic_pressure_variable as compute_dynamic_pressure_variable,
)
from pyvl.postprocess.pressure import (
    compute_surface_dynamic_pressure as compute_surface_dynamic_pressure,
)

# Velocity Field
from pyvl.postprocess.velocity import compute_velocities as compute_velocities
from pyvl.postprocess.velocity import (
    compute_velocities_variable as compute_velocities_variable,
)
