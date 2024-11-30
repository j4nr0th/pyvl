"""Implementation of Geometry related operations."""

from collections.abc import Iterable
from dataclasses import dataclass

import meshio as mio
import numpy as np
import numpy.typing as npt
import pyvista as pv

from pydust.cdust import Mesh, ReferenceFrame


@dataclass(init=False, frozen=True)
class Geometry:
    """Class which describes a geometry compoenent."""

    label: str
    reference_frame: ReferenceFrame
    primal: Mesh
    dual: Mesh

    def __init__(
        self, label: str, reference_frame: ReferenceFrame, mesh: mio.Mesh | Mesh
    ) -> None:
        if not isinstance(label, str):
            raise TypeError(
                f"label must be a string, instead it was {type(label).__name__}."
            )
        if not isinstance(reference_frame, ReferenceFrame):
            raise TypeError(
                "reference_frame must be a ReferenceFrame, instead it was "
                f"{type(reference_frame).__name__}."
            )
        if not isinstance(mesh, (Mesh, mio.Mesh)):
            raise TypeError(
                f"mesh must be a either {Mesh} or {mio.Mesh}, instead it was "
                f"{type(mesh)}."
            )
        if isinstance(mesh, mio.Mesh):
            connections: list[npt.NDArray[np.unsignedinteger]] = []
            c: mio.CellBlock
            for c in mesh.cells:
                if c.dim != 2:
                    raise ValueError(
                        "The mesh contains a cell block, which has topological "
                        "dimesion not equal to 2"
                    )
                for element in c.data:
                    connections.append(np.asarray(element, np.uint32))
            mesh = Mesh(mesh.points, connections)

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "reference_frame", reference_frame)
        object.__setattr__(self, "primal", mesh)
        object.__setattr__(self, "dual", mesh.compute_dual())

    def as_polydata(self) -> pv.PolyData:
        """Convert geometry into PyVista's PolyData."""
        positions = self.reference_frame.from_parent_with_offset(self.primal.positions)
        nper_elem, indices = self.primal.to_element_connectivity()
        offsets = np.pad(np.cumsum(nper_elem), (1, 0))
        faces = [indices[offsets[i] : offsets[i + 1]] for i in range(nper_elem.size)]
        pd = pv.PolyData.from_irregular_faces(positions, faces)
        return pd


def geometry_show_pyvista(
    geometries: Iterable[Geometry], plt: pv.Plotter | None = None
) -> None:
    """Show the geometry using PyVista."""
    show = plt is None
    if plt is None:
        plt = pv.Plotter(theme=pv.themes.DocumentProTheme())
        plt.theme.show_edges = True
        plt.theme.show_scalar_bar = False

    for geo in geometries:
        plt.add_mesh(geo.as_polydata(), label=geo.label)

    if show:
        plt.show()
