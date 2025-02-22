"""Tests for serialization to and from HDF5 files."""

import warnings
from pathlib import Path

import meshio as mio
import numpy as np
from pyvl import Geometry, ReferenceFrame, TranslatingReferenceFrame, mesh_from_mesh_io
from pyvl import fio as fio
from pyvl.geometry import rf_from_serial, rf_to_serial


def test_rf_serialization1() -> None:
    """Check if ReferenceFrame is properly serialized."""
    np.random.seed(0)

    rf = ReferenceFrame(
        np.random.random_sample(3),
        np.random.random_sample(3),
        None,
    )

    out = rf_to_serial(rf)
    rf1 = rf_from_serial(out)
    assert all(rf.offset == rf1.offset)
    assert all(rf.angles == rf1.angles)
    assert rf.parent == rf1.parent


def test_rf_serialization2() -> None:
    """Check if ReferenceFrame with parent is properly serialized."""
    np.random.seed(0)

    rf_1 = ReferenceFrame(
        np.random.random_sample(3),
        np.random.random_sample(3),
        None,
    )
    rf_2 = ReferenceFrame(
        np.random.random_sample(3),
        np.random.random_sample(3),
        rf_1,
    )

    out = rf_to_serial(rf_2)

    rf_in = rf_from_serial(out)
    assert all(rf_in.offset == rf_2.offset)
    assert all(rf_in.angles == rf_2.angles)
    assert rf_in.parent is not None
    assert rf_2.parent is not None
    assert all(rf_in.parent.offset == rf_2.parent.offset)
    assert all(rf_in.parent.angles == rf_2.parent.angles)
    assert rf_in.parent.parent == rf_2.parent.parent


def test_rf_serialization3() -> None:
    """Check if ReferenceFrame of different types."""
    np.random.seed(0)

    rf_1 = TranslatingReferenceFrame(
        np.random.random_sample(3),
        np.random.random_sample(3),
        np.random.random_sample(3),
        None,
        np.random.random_sample(1)[0],
    )
    rf_2 = TranslatingReferenceFrame(
        np.random.random_sample(3),
        np.random.random_sample(3),
        np.random.random_sample(3),
        rf_1,
        np.random.random_sample(1)[0],
    )

    out = rf_to_serial(rf_2)

    rf_in = rf_from_serial(out)
    assert all(rf_in.offset == rf_2.offset)
    assert all(rf_in.angles == rf_2.angles)
    assert rf_in.parent is not None
    assert rf_2.parent is not None
    assert all(rf_in.parent.offset == rf_2.parent.offset)
    assert all(rf_in.parent.angles == rf_2.parent.angles)
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

    out = geo.save()

    geo1 = Geometry.load(geo.label, out)
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

    out = geo.save()
    fio.serialize_hdf5(out, fpath)
    inv = fio.deserialize_hdf5(fpath)

    geo1 = Geometry.load(geo.label, inv)
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

    out = geo.save()
    fio.serialize_json(out, fpath)
    inv = fio.deserialize_json(fpath)

    geo1 = Geometry.load(geo.label, inv)
    assert geo1.label == geo.label
    assert np.all(geo1.positions == geo.positions)
    assert geo1.msh == geo.msh
