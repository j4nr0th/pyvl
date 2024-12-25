"""Implementation of the HDF5 based IO."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pyvl.fio.io_common import (
    HirearchicalMap,
)


class _CustomJSONEncoder(json.JSONEncoder):
    """Custom encoder to handle HirearchicalMap objects."""

    def default(self, o: Any) -> Any:
        """Return a seriazible object."""
        if isinstance(o, HirearchicalMap):
            return o._map
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def serialize_json(hmap: HirearchicalMap, path: Path | str) -> None:
    """Save a HirearchicalMap into a JSON file."""
    encoder = _CustomJSONEncoder()
    with open(path, "w") as f_out:
        f_out.write(encoder.encode(hmap))


def _deserialize_function(arg: dict[Any, Any]) -> HirearchicalMap:
    """Deserialized into a HirearchicalMap."""
    hm = HirearchicalMap()
    for key in arg:
        v = arg[key]
        if isinstance(v, Sequence) and not isinstance(v, str):
            hm[key] = np.array(v)
        else:
            hm[key] = v
    return hm


def deserialize_json(path: Path | str) -> HirearchicalMap:
    """Load a HirearchicalMap from a JSON file."""
    with open(path, "r") as f_in:
        return json.load(f_in, object_hook=_deserialize_function)
