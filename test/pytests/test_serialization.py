"""Tests for serialization to and from HDF5 files."""

import warnings
from pathlib import Path

import meshio as mio
import numpy as np
from pyvl import Geometry, ReferenceFrame, fio, mesh_from_mesh_io
from pyvl.fio.io_common import PythonSerializer
from pyvl.geometry import rf_to_serial


def test_rf_serialization1() -> None:
    """Check if ReferenceFrame is properly serialized."""
    rng = np.random.default_rng(0)
    serializer = PythonSerializer()

    rf = ReferenceFrame(
        rng.random(3),
        rng.random(3),
        rng.random(3),
        rng.random(3),
        None,
    )

    out = rf_to_serial(rf, serializer.serialize)
    rf1 = ReferenceFrame.load(out, serializer.deserialize)
    assert all(rf.offset_at() == rf1.offset_at())
    assert all(rf.angles_at() == rf1.angles_at())
    assert all(rf.velocity_at() == rf1.velocity_at())
    assert all(rf.rotation_at() == rf1.rotation_at())
    assert rf.parent == rf1.parent


def test_rf_serialization2() -> None:
    """Check if ReferenceFrame with parent is properly serialized."""
    rng = np.random.default_rng(0)

    rf_1 = ReferenceFrame(
        rng.random(3),
        rng.random(3),
        rng.random(3),
        rng.random(3),
    )
    rf_2 = ReferenceFrame(
        rng.random(3),
        rng.random(3),
        rng.random(3),
        rng.random(3),
        parent=rf_1,
    )

    serializer = PythonSerializer()
    out = rf_to_serial(rf_2, serializer.serialize)

    rf_in = ReferenceFrame.load(out, serializer.deserialize)
    assert all(rf_in.offset_at() == rf_2.offset_at())
    assert all(rf_in.angles_at() == rf_2.angles_at())
    assert all(rf_in.velocity_at() == rf_2.velocity_at())
    assert all(rf_in.rotation_at() == rf_2.rotation_at())
    assert rf_in.parent is not None
    assert rf_2.parent is not None
    assert all(rf_in.parent.offset_at() == rf_2.parent.offset_at())
    assert all(rf_in.parent.angles_at() == rf_2.parent.angles_at())
    assert all(rf_in.parent.velocity_at() == rf_2.parent.velocity_at())
    assert all(rf_in.parent.rotation_at() == rf_2.parent.rotation_at())
    assert rf_in.parent.parent == rf_2.parent.parent


def test_geometry_serialization1() -> None:
    """Test that geometry is properly serialized and de-serialized."""
    m = mio.read("test/pytests/test_inputs/cylinder.msh")
    with warnings.catch_warnings(action="ignore", category=UserWarning):
        pos, msh = mesh_from_mesh_io(m)

    geo = Geometry(
        "test_geometry",
        ReferenceFrame(),
        msh,
        pos,
    )

    serializer = PythonSerializer()
    out = geo.save(serializer.serialize)

    geo1 = Geometry.load(out, serializer.deserialize)
    assert geo1.label == geo.label
    assert np.all(geo1.positions == geo.positions)
    assert geo1.msh == geo.msh


def test_geometry_serialization_hdf() -> None:
    """Test that geometry is properly serialized and de-serialized with HDF5."""
    m = mio.read("test/pytests/test_inputs/cylinder.msh")
    fpath = Path("test/pytests/test_outputs/ser_geo_1.h5")
    with warnings.catch_warnings(action="ignore", category=UserWarning):
        pos, msh = mesh_from_mesh_io(m)

    geo = Geometry(
        "test_geometry",
        ReferenceFrame(),
        msh,
        pos,
    )

    serializer = PythonSerializer()
    out = geo.save(serializer.serialize)
    fio.serialize_hdf5(out, fpath)
    inv = fio.deserialize_hdf5(fpath)

    geo1 = Geometry.load(inv, serializer.deserialize)
    assert geo1.label == geo.label
    assert np.all(geo1.positions == geo.positions)
    assert geo1.msh == geo.msh


def test_geometry_serialization_json() -> None:
    """Test that geometry is properly serialized and de-serialized with JSON."""
    m = mio.read("test/pytests/test_inputs/cylinder.msh")
    fpath = Path("test/pytests/test_outputs/ser_geo_1.json")
    with warnings.catch_warnings(action="ignore", category=UserWarning):
        pos, msh = mesh_from_mesh_io(m)

    geo = Geometry(
        "test_geometry",
        ReferenceFrame(),
        msh,
        pos,
    )

    serializer = PythonSerializer()
    out = geo.save(serializer.serialize)
    fio.serialize_json(out, fpath)
    inv = fio.deserialize_json(fpath)

    geo1 = Geometry.load(inv, serializer.deserialize)
    assert geo1.label == geo.label
    assert np.all(geo1.positions == geo.positions)
    assert geo1.msh == geo.msh
