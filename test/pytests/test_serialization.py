"""Tests for serialization to and from HDF5 files."""

import warnings
from pathlib import Path

import meshio as mio
import numpy as np
import pytest
from pyvl import Geometry, ReferenceFrame, mesh_from_mesh_io
from pyvl import fio as fio
from pyvl.fio.io_common import HirearchicalMap
from pyvl.fio.type_resolution import (
    flow_conditions_from_serial,
    reference_frame_from_serial,
    wake_model_from_serial,
)
from pyvl.flow_conditions import FlowConditionsUniform
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
    assert all(rf.offset_at() == rf1.offset_at())
    assert all(rf.angles_at() == rf1.angles_at())
    assert rf.parent == rf1.parent


def test_rf_serialization2() -> None:
    """Check if ReferenceFrame with parent is properly serialized."""
    np.random.seed(0)

    rf_1 = ReferenceFrame(
        np.random.random_sample(3),
        np.random.random_sample(3),
    )
    rf_2 = ReferenceFrame(
        np.random.random_sample(3),
        np.random.random_sample(3),
        parent=rf_1,
    )

    out = rf_to_serial(rf_2)

    rf_in = rf_from_serial(out)
    assert all(rf_in.offset_at() == rf_2.offset_at())
    assert all(rf_in.angles_at() == rf_2.angles_at())
    assert rf_in.parent is not None
    assert rf_2.parent is not None
    assert all(rf_in.parent.offset_at() == rf_2.parent.offset_at())
    assert all(rf_in.parent.angles_at() == rf_2.parent.angles_at())
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


def test_rf_from_serial_unknown_type() -> None:
    """Test that unknown ReferenceFrame type raises TypeError."""
    group = HirearchicalMap()
    group.insert_string("type", "pyvl.nonexistent.CustomReferenceFrame")
    data = HirearchicalMap()
    group.insert_hirearchycal_map("data", data)

    with pytest.raises(TypeError, match="Unknown ReferenceFrame type"):
        reference_frame_from_serial(group)


def test_rf_from_serial_custom_types() -> None:
    """Test that custom types can be registered for ReferenceFrame."""

    class CustomReferenceFrame(ReferenceFrame):
        pass

    data = HirearchicalMap()
    ReferenceFrame().save(data)
    group = HirearchicalMap()
    group.insert_string("type", "my.custom.ReferenceFrame")
    group.insert_hirearchycal_map("data", data)

    custom_types = {"my.custom.ReferenceFrame": CustomReferenceFrame}
    with pytest.raises(
        TypeError, match="is registered in custom_types but allow_override is False"
    ):
        reference_frame_from_serial(group, custom_types, allow_override=False)

    rf = reference_frame_from_serial(group, custom_types, allow_override=True)
    assert isinstance(rf, CustomReferenceFrame)


def test_rf_from_serial_custom_override() -> None:
    """Test that custom types can override built-in types when allow_override=True."""

    class CustomReferenceFrame(ReferenceFrame):
        pass

    group = HirearchicalMap()
    group.insert_string("type", "pyvl.cvl.ReferenceFrame")
    data = HirearchicalMap()
    ReferenceFrame().save(data)
    group.insert_hirearchycal_map("data", data)

    custom_types = {"pyvl.cvl.ReferenceFrame": CustomReferenceFrame}
    with pytest.raises(
        TypeError, match="is registered in custom_types but allow_override is False"
    ):
        reference_frame_from_serial(group, custom_types, allow_override=False)

    rf = reference_frame_from_serial(group, custom_types, allow_override=True)
    assert isinstance(rf, CustomReferenceFrame)


def test_flow_conditions_unknown_type() -> None:
    """Test that unknown FlowConditions type raises TypeError."""
    group = HirearchicalMap()
    group.insert_string("type", "pyvl.nonexistent.CustomFlowConditions")
    data = HirearchicalMap()
    group.insert_hirearchycal_map("data", data)

    with pytest.raises(TypeError, match="Unknown FlowConditions type"):
        flow_conditions_from_serial(group)


def test_flow_conditions_custom_types() -> None:
    """Test that custom types can be registered for FlowConditions."""

    class CustomFlowConditions(FlowConditionsUniform):
        pass

    group = HirearchicalMap()
    group.insert_string("type", "my.custom.FlowConditions")
    data = FlowConditionsUniform(1.0, 2.0, 3.0).save()
    group.insert_hirearchycal_map("data", data)

    custom_types = {"my.custom.FlowConditions": CustomFlowConditions}
    with pytest.raises(
        TypeError, match="is registered in custom_types but allow_override is False"
    ):
        flow_conditions_from_serial(group, custom_types, allow_override=False)

    fc = flow_conditions_from_serial(group, custom_types, allow_override=True)
    assert isinstance(fc, CustomFlowConditions)


def test_flow_conditions_override() -> None:
    """Test that custom types can override built-in FlowConditions."""

    class CustomFlowConditions(FlowConditionsUniform):
        pass

    group = HirearchicalMap()
    group.insert_string("type", "pyvl.flow_conditions.FlowConditionsUniform")
    data = FlowConditionsUniform(1.0, 2.0, 3.0).save()
    group.insert_hirearchycal_map("data", data)

    custom_types = {"pyvl.flow_conditions.FlowConditionsUniform": CustomFlowConditions}
    with pytest.raises(
        TypeError, match="is registered in custom_types but allow_override is False"
    ):
        flow_conditions_from_serial(group, custom_types, allow_override=False)

    fc = flow_conditions_from_serial(group, custom_types, allow_override=True)
    assert isinstance(fc, CustomFlowConditions)


def test_wake_model_unknown_type() -> None:
    """Test that unknown WakeModel type raises TypeError."""
    group = HirearchicalMap()
    group.insert_string("type", "pyvl.nonexistent.CustomWakeModel")
    data = HirearchicalMap()
    group.insert_hirearchycal_map("data", data)

    with pytest.raises(TypeError, match="Unknown WakeModel type"):
        wake_model_from_serial(group)
