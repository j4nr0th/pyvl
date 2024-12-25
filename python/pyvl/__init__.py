"""PyVL is a package used for potential flow analysis for geometries."""

# C types
from pyvl.cvl import GeoID as GeoID
from pyvl.cvl import Line as Line
from pyvl.cvl import Mesh as Mesh
from pyvl.cvl import ReferenceFrame as ReferenceFrame
from pyvl.cvl import Surface as Surface

# IO
from pyvl.fio.io_common import HirearchicalMap as HirearchicalMap

# Flow Conditions
from pyvl.flow_conditions import FlowConditions as FlowConditions
from pyvl.flow_conditions import FlowConditionsRotating as FlowConditionsRotating
from pyvl.flow_conditions import FlowConditionsUniform as FlowConditionsUniform

# Geometry
from pyvl.geometry import Geometry as Geometry
from pyvl.geometry import SimulationGeometry as SimulationGeometry
from pyvl.geometry import geometry_show_pyvista as geometry_show_pyvista
from pyvl.geometry import mesh_from_mesh_io as mesh_from_mesh_io

# Reference Frames
from pyvl.reference_frames import RotorReferenceFrame as RotorReferenceFrame
from pyvl.reference_frames import TranslatingReferenceFrame as TranslatingReferenceFrame

# Settings
from pyvl.settings import ModelSettings as ModelSettings
from pyvl.settings import SolverSettings as SolverSettings
from pyvl.settings import TimeSettings as TimeSettings

# Solver
from pyvl.solver import compute_induced_velocities as compute_induced_velocities
from pyvl.solver import run_solver as run_solver

# Wake models
from pyvl.wake import WakeModel as WakeModel
from pyvl.wake_models import (
    WakeModelLineExplicitUnsteady as WakeModelLineExplicitUnsteady,
)
