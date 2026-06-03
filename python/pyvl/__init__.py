"""PyVL is a package used for potential flow analysis for geometries."""

# Examples
from pyvl import examples as examples

# Post processing
from pyvl import postprocess as postprocess

# C types
from pyvl.cvl import GeoID as GeoID
from pyvl.cvl import Mesh as Mesh
from pyvl.cvl import ReferenceFrame as ReferenceFrame

# File IO
from pyvl.fio.io_common import HirearchicalMap as HirearchicalMap
from pyvl.fio.io_common import PythonSerializer as PythonSerializer

# Flow Conditions
from pyvl.flow_conditions import FlowConditions as FlowConditions
from pyvl.flow_conditions import FlowConditionsRotating as FlowConditionsRotating
from pyvl.flow_conditions import FlowConditionsUniform as FlowConditionsUniform

# Geometry
from pyvl.geometry import Geometry as Geometry
from pyvl.geometry import SimulationGeometry as SimulationGeometry
from pyvl.geometry import geometry_show_pyvista as geometry_show_pyvista
from pyvl.geometry import mesh_from_mesh_io as mesh_from_mesh_io

# Settings
from pyvl.settings import ModelSettings as ModelSettings
from pyvl.settings import SolverSettings as SolverSettings
from pyvl.settings import TimeSettings as TimeSettings

# Solver
from pyvl.solver import OutputSettings as OutputSettings
from pyvl.solver import run_solver as run_solver

# Wake
from pyvl.wake import WakeState as WakeState
