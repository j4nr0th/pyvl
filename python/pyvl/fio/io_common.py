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

    def insert_array(self, key: str, value: npt.ArrayLike) -> None:
        """Insert an array-like into the mapping and copies it."""
        self._insert(key, np.array(value))

    def insert_string(self, key: str, value: str) -> None:
        """Insert a string into the mapping."""
        if not isinstance(value, str):
            raise TypeError(f"The value was not a string but {type(value).__name__}")
        self._insert(key, value)

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

    def get_array(self, key: str) -> npt.NDArray:
        """Load a copy of an array from the mapping."""
        v = self[key]
        return np.array(v)

    def get_string(self, key: str) -> str:
        """Load a string from the mapping."""
        value = self._map[key]
        if isinstance(value, str):
            return value
        if isinstance(value, np.ndarray):
            return str(value)
        raise TypeError(f"The value was not a string but {type(value).__name__}")

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


SerializationFunction = Callable[[HirearchicalMap, Path | str], None]
DeserializationFunction = Callable[[Path | str], HirearchicalMap]


class PythonSerializer:
    """Serializer for the current Python session."""

    _contents: dict[str, Callable]

    def __init__(self) -> None:
        self._contents = dict()

    def serialize(self, fn: Callable) -> str:
        """Serialize a string to a callable."""
        try:
            label = fn.__name__
        except Exception as e:
            del e
            label = f"anonymous_{len(self._contents)}"
        if label not in self._contents:
            self._contents[label] = fn
        return label

    def deserialize(self, key: str) -> Callable:
        """Deserialize a callable based on the label."""
        return self._contents[key]
