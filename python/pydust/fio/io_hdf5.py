"""Implementation of the HDF5 based IO."""

from collections.abc import Iterator
from pathlib import Path

import h5py

from pydust.fio.io_common import (
    HirearchicalMap,
)


def serialize_hdf5(hmap: HirearchicalMap, path: Path | str) -> None:
    """Save a HirearchicalMap into a HDF5 file."""
    with h5py.File(path, "w") as f_out:
        iterators: list[tuple[Iterator[str], HirearchicalMap, h5py.Group | h5py.File]] = [
            (iter(hmap), hmap, f_out)
        ]
        while iterators:
            it, hm, out = iterators.pop()
            for key in it:
                val = hm[key]
                if isinstance(val, HirearchicalMap):
                    iterators.append((it, hm, out))
                    iterators.append((iter(val), val, out.create_group(key)))
                    break
                out[key] = val


def deserialize_hdf5(path: Path | str) -> HirearchicalMap:
    """Load a HirearchicalMap from a HDF5 file."""
    hmap = HirearchicalMap()
    with h5py.File(path, "r") as f_in:
        iterators: list[tuple[Iterator, HirearchicalMap, h5py.Group | h5py.File]] = [
            (iter(f_in), hmap, f_in)
        ]
        while iterators:
            it, hm, src = iterators.pop()
            for key in it:
                val = src[key]
                if isinstance(val, h5py.Group):
                    iterators.append((it, hm, src))
                    new_hm = HirearchicalMap()
                    hm.insert_hirearchycal_map(key, new_hm)
                    iterators.append((iter(val), new_hm, val))
                    break
                assert not isinstance(val, h5py.Datatype)
                v = val[()]
                if isinstance(v, bytes):
                    v = v.decode()
                hm[key] = v
    return hmap
