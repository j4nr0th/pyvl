"""Tests for serialization to and from HDF5 files."""

import h5py
import meshio as mio
import numpy as np
from pydust import Geometry, ReferenceFrame, TranslatingReferenceFrame, mesh_from_mesh_io
from pydust.geometry import rf_from_hdf5, rf_to_hdf5


def test_rf_serialization1() -> None:
    """Check if ReferenceFrame is properly serialized."""
    np.random.seed(0)
    fpath = "test/pytests/test_outputs/ser_rf_1.h5"

    rf = ReferenceFrame(
        np.random.random_sample(3),
        np.random.random_sample(3),
        None,
    )

    with h5py.File(fpath, "w") as f_out:
        out_group = f_out.create_group("test_name")
        rf_to_hdf5(rf, out_group)

    with h5py.File(fpath, "r") as f_in:
        in_group = f_in["test_name"]
        assert isinstance(in_group, h5py.Group)
        rf1 = rf_from_hdf5(in_group)
        assert all(rf.offset == rf1.offset)
        assert all(rf.angles == rf1.angles)
        assert rf.parent == rf1.parent


def test_rf_serialization2() -> None:
    """Check if ReferenceFrame with parent is properly serialized."""
    np.random.seed(0)
    fpath = "test/pytests/test_outputs/ser_rf_2.h5"

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

    with h5py.File(fpath, "w") as f_out:
        out_group = f_out.create_group("test_name")
        rf_to_hdf5(rf_2, out_group)

    with h5py.File(fpath, "r") as f_in:
        in_group = f_in["test_name"]
        assert isinstance(in_group, h5py.Group)
        rf_in = rf_from_hdf5(in_group)
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
    fpath = "test/pytests/test_outputs/ser_rf_3.h5"

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

    with h5py.File(fpath, "w") as f_out:
        out_group = f_out.create_group("test_name")
        rf_to_hdf5(rf_2, out_group)

    with h5py.File(fpath, "r") as f_in:
        in_group = f_in["test_name"]
        assert isinstance(in_group, h5py.Group)
        rf_in = rf_from_hdf5(in_group)
        assert all(rf_in.offset == rf_2.offset)
        assert all(rf_in.angles == rf_2.angles)
        assert rf_in.parent is not None
        assert rf_2.parent is not None
        assert all(rf_in.parent.offset == rf_2.parent.offset)
        assert all(rf_in.parent.angles == rf_2.parent.angles)
        assert rf_in.parent.parent == rf_2.parent.parent


def test_geometry_serialization1() -> None:
    """Test that geometry is properly serialized and de-serialized."""
    fpath = "test/pytests/test_outputs/ser_geo_1.h5"
    m = mio.read("test/pytests/test_inputs/cylinder.msh")
    pos, msh = mesh_from_mesh_io(m)

    geo = Geometry(
        "test_geometry",
        ReferenceFrame(),
        msh,
        pos,
    )

    with h5py.File(fpath, "w") as f_out:
        out_group = f_out.create_group(geo.label)
        geo.save(out_group)

    with h5py.File(fpath, "r") as f_in:
        key = [k for k in f_in.keys()]
        assert len(key) == 1
        label = key[0]
        in_group = f_in[label]
        assert isinstance(in_group, h5py.Group)
        geo1 = Geometry.load(label, in_group)
        assert geo1.label == geo.label
        assert np.all(geo1.positions == geo.positions)
        assert geo1.msh == geo.msh  # TODO


if __name__ == "__main__":
    test_geometry_serialization1()
