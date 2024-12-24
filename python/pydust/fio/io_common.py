"""Classes and implementation of IO related functionality."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from numpy import typing as npt


class HirearchicalMap(MutableMapping[str, Any]):
    """Mapping which contains other hierarchical mappings or values uniquly."""

    _map: dict[str, HirearchicalMap | Any]

    def __init__(self) -> None:
        self._map = dict()

    def __getitem__(self, key: str) -> Any:
        """Return the value associated with the key."""
        return self._map[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Add a value associated with the key."""
        self._map[key] = value

    def insert_type(self, key: str, t: type) -> None:
        """Insert a type into the mapping as an entry "type"."""
        if not isinstance(t, type):
            raise TypeError(f"The value was not a type but {type(t).__name__}")
        self._insert(key, t.__module__ + "." + t.__name__)

    def insert_array(self, key: str, value: npt.ArrayLike) -> None:
        """Insert an array-like into the mapping and copies it."""
        self._insert(key, np.array(value))

    def insert_string(self, key: str, value: str) -> None:
        """Insert a string into the mapping."""
        if not isinstance(value, str):
            raise TypeError(f"The value was not a string but {type(value).__name__}")
        self._insert(key, np.array(value))

    def insert_scalar(self, key: str, value: int | float) -> None:
        """Insert a scalar into the mapping."""
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"The value was not an int or float but {type(value).__name__}"
            )
        self._insert(key, value)

    def insert_int(self, key: str, value: int) -> None:
        """Insert a int into the mapping."""
        if not isinstance(value, int):
            raise TypeError(f"The value was not an int but {type(value).__name__}")
        self._insert(key, value)

    def _recursion_check(self, value: HirearchicalMap) -> bool:
        """Check if the value would cause a recursive hirearchiy."""
        for k in self._map:
            v = self._map[k]
            if not isinstance(v, HirearchicalMap):
                continue
            if v._recursion_check(value):
                return True

        return False

    def insert_hirearchycal_map(self, key: str, value: HirearchicalMap) -> None:
        """Insert another mapping into the mapping."""
        if self._recursion_check(value):
            raise ValueError(
                "Inserting the hierarchical map would cause cyclical hierarchy."
            )
        if not isinstance(value, HirearchicalMap):
            raise TypeError(
                f"The value was not a HirearchicalMap but {type(value).__name__}"
            )
        self._insert(key, value)

    def get_type(self, key: str) -> type:
        """Load a type from the mapping as an entry "type"."""
        full_type_name = self[key]
        if not isinstance(full_type_name, str):
            raise TypeError(
                "The value was not a type name but instead "
                f"{type(full_type_name).__name__}"
            )
        module_name, type_name = full_type_name.rsplit(".", 1)
        mod = __import__(module_name, fromlist=[type_name])
        cls: type = getattr(mod, type_name)
        if not isinstance(cls, type):
            raise TypeError(f"The value was not a type but {type(cls).__name__}")
        return cls

    def get_array(self, key: str) -> npt.NDArray:
        """Load a copy of an array from the mapping."""
        v = self[key]
        return np.array(v)

    def get_string(self, key: str) -> str:
        """Load a string from the mapping."""
        value = self._map[key]
        if not isinstance(value, str):
            raise TypeError(f"The value was not a strint but {type(value).__name__}")
        return value

    def get_scalar(self, key: str) -> int | float:
        """Load a scalar from the mapping."""
        value = self._map[key]
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"The value was not an int or float but {type(value).__name__}"
            )
        return value

    def get_int(self, key: str) -> int:
        """Load a scalar from the mapping."""
        value = self._map[key]
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"The value was not an int but {type(value).__name__}")
        return int(value)

    def get_hirearchical_map(self, key: str) -> HirearchicalMap:
        """Load a hierarchical map from the mapping."""
        value = self._map[key]
        if not isinstance(value, HirearchicalMap):
            raise TypeError(
                f"The value was not a HirearchicalMap but {type(value).__name__}"
            )

        return value

    def _insert(self, key: str, value: Any) -> None:
        """Set the value associated with the key."""
        if not isinstance(key, str):
            raise TypeError(f"Key is not a string but a {type(key).__name__}.")
        if key in self._map:
            raise KeyError(f'Map already contains a key "{key}".')
        self._map[key] = value

    def __len__(self) -> int:
        """Return the number of key-value pairs in the mapping."""
        return len(self._map)

    def __delitem__(self, key: str) -> None:
        """Remove the item from the mapping."""
        del self._map[key]

    def __iter__(self) -> Iterator[str]:
        """Return iterator over keys."""
        return iter(self._map)


SerializationFunction = Callable[[HirearchicalMap, Path], None]
DeserializationFunction = Callable[[Path], HirearchicalMap]
