"""PyVL is a package used for potential flow analysis for geometries."""

# Examples
from pyvl import examples as examples

# Mesh utilities
from pyvl import meshing as meshing

# Post processing
from pyvl import postprocess as postprocess

# C types
from pyvl.cvl import INVALID_ID as INVALID_ID
from pyvl.cvl import BarnesHutTree as BarnesHutTree
from pyvl.cvl import GeoID as GeoID
from pyvl.cvl import Mesh as Mesh
from pyvl.cvl import Multipole as Multipole
from pyvl.cvl import ReferenceFrame as ReferenceFrame
from pyvl.cvl import TransformationPlane as TransformationPlane

# File IO
from pyvl.fio.io_common import HirearchicalMap as HirearchicalMap
from pyvl.fio.io_common import PredefinedSerializer as PredefinedSerializer
from pyvl.fio.io_common import PythonSerializer as PythonSerializer

# Geometry
from pyvl.geometry import Geometry as Geometry
from pyvl.geometry import SimulationGeometry as SimulationGeometry
from pyvl.geometry import geometry_show_pyvista as geometry_show_pyvista
from pyvl.geometry import mesh_from_mesh_io as mesh_from_mesh_io

# Settings
from pyvl.settings import ModelSettings as ModelSettings
from pyvl.settings import SolverSettings as SolverSettings
from pyvl.settings import WakeSettings as WakeSettings
from pyvl.settings import WakeShedderCallback as WakeShedderCallback
from pyvl.settings import WakeShedderUniform as WakeShedderUniform

# Solver
from pyvl.solver import OutputSettings as OutputSettings
from pyvl.solver import SolverState as SolverState
from pyvl.solver import run_solver as run_solver
from pyvl.solver import run_solver_steady_state as run_solver_steady_state
from pyvl.solver import update_simulation_state as update_simulation_state

# Wake
from pyvl.wake import WakeState as WakeState
