"""Submodule which has implementation of IO functionality."""

# Common
from pyvl.fio.io_common import HirearchicalMap as HirearchicalMap

# HDF5
from pyvl.fio.io_hdf5 import deserialize_hdf5 as deserialize_hdf5
from pyvl.fio.io_hdf5 import serialize_hdf5 as serialize_hdf5

# JSON
from pyvl.fio.io_json import deserialize_json as deserialize_json
from pyvl.fio.io_json import serialize_json as serialize_json
