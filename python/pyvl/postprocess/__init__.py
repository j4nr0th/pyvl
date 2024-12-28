"""Submodule with implementation of post processing analysis."""

# Velocity Field
from pyvl.postprocess.forces import circulatory_forces as circulatory_forces

# Force Field
from pyvl.postprocess.velocity import compute_velocities as compute_velocities
from pyvl.postprocess.velocity import (
    compute_velocities_variable as compute_velocities_variable,
)
