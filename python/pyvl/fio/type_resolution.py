"""Type resolution functions using match-case for serialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pyvl.fio.io_common import HirearchicalMap

if TYPE_CHECKING:
    from pyvl.cvl import ReferenceFrame
    from pyvl.flow_conditions import FlowConditions
    from pyvl.wake import WakeModel


def reference_frame_from_serial(
    group: HirearchicalMap,
    custom_types: Mapping[str, type] | None = None,
    allow_override: bool = False,
) -> ReferenceFrame:
    """Load a ReferenceFrame from a HirearchicalMap.

    This function uses match-case to determine the appropriate ReferenceFrame
    subclass to instantiate. Custom types can be registered via the ``custom_types``
    mapping.

    Parameters
    ----------
    group : HirearchicalMap
        The serialized data containing the ReferenceFrame information.
    custom_types : Mapping[str, type], optional
        A mapping of type names to types, allowing registration of custom
        ReferenceFrame subclasses. If ``allow_override`` is True, these can
        override the built-in types.
    allow_override : bool, default: False
        If True, custom types from ``custom_types`` will be checked first and
        can override built-in types. If False, custom types are only used when
        the type name is not a known built-in type.

    Returns
    -------
    ReferenceFrame
        The deserialized ReferenceFrame object.

    Raises
    ------
    TypeError
        If the type name is not recognized or cannot be loaded.
    """
    from pyvl.cvl import ReferenceFrame

    type_name = group.get_string("type")
    data = group.get_hirearchical_map("data")
    parent = None
    if "parent" in group:
        parent_group = group.get_hirearchical_map("parent")
        parent = reference_frame_from_serial(parent_group, custom_types, allow_override)

    match type_name:
        case _ if custom_types is not None and type_name in custom_types:
            if allow_override:
                cls = custom_types[type_name]
                return cls.load(data, parent) if parent else cls.load(data)
            raise TypeError(
                f'Type "{type_name}" is registered in custom_types but '
                "allow_override is False."
            )
        case "pyvl.cvl.ReferenceFrame":
            return (
                ReferenceFrame.load(data, parent) if parent else ReferenceFrame.load(data)
            )
        case _:
            if custom_types is not None:
                cls = custom_types.get(type_name)
                if cls is not None:
                    return cls.load(data, parent) if parent else cls.load(data)
            raise TypeError(
                f'Unknown ReferenceFrame type "{type_name}". '
                "Provide custom_types mapping or check type name."
            )


def flow_conditions_from_serial(
    group: HirearchicalMap,
    custom_types: Mapping[str, type] | None = None,
    allow_override: bool = False,
) -> FlowConditions:
    """Load FlowConditions from a HirearchicalMap.

    This function uses match-case to determine the appropriate FlowConditions
    subclass to instantiate. Custom types can be registered via the ``custom_types``
    mapping.

    Parameters
    ----------
    group : HirearchicalMap
        The serialized data containing the FlowConditions information.
    custom_types : Mapping[str, type], optional
        A mapping of type names to types, allowing registration of custom
        FlowConditions subclasses. If ``allow_override`` is True, these can
        override the built-in types.
    allow_override : bool, default: False
        If True, custom types from ``custom_types`` will be checked first and
        can override built-in types. If False, custom types are only used when
        the type name is not a known built-in type.

    Returns
    -------
    FlowConditions
        The deserialized FlowConditions object.

    Raises
    ------
    TypeError
        If the type name is not recognized or cannot be loaded.
    """
    from pyvl.flow_conditions import FlowConditionsRotating, FlowConditionsUniform

    type_name = group.get_string("type")
    data = group.get_hirearchical_map("data")

    match type_name:
        case _ if custom_types is not None and type_name in custom_types:
            if allow_override:
                cls = custom_types[type_name]
                return cls.load(data)
            raise TypeError(
                f'Type "{type_name}" is registered in custom_types but '
                "allow_override is False."
            )
        case "pyvl.flow_conditions.FlowConditionsUniform":
            return FlowConditionsUniform.load(data)
        case "pyvl.flow_conditions.FlowConditionsRotating":
            return FlowConditionsRotating.load(data)
        case _:
            if custom_types is not None:
                cls = custom_types.get(type_name)
                if cls is not None:
                    return cls.load(data)
            raise TypeError(
                f'Unknown FlowConditions type "{type_name}". '
                "Provide custom_types mapping or check type name."
            )


def wake_model_from_serial(
    group: HirearchicalMap,
    custom_types: Mapping[str, type] | None = None,
    allow_override: bool = False,
) -> WakeModel:
    """Load WakeModel from a HirearchicalMap.

    This function uses match-case to determine the appropriate WakeModel
    subclass to instantiate. Custom types can be registered via the ``custom_types``
    mapping.

    Parameters
    ----------
    group : HirearchicalMap
        The serialized data containing the WakeModel information.
    custom_types : Mapping[str, type], optional
        A mapping of type names to types, allowing registration of custom
        WakeModel subclasses. If ``allow_override`` is True, these can
        override the built-in types.
    allow_override : bool, default: False
        If True, custom types from ``custom_types`` will be checked first and
        can override built-in types. If False, custom types are only used when
        the type name is not a known built-in type.

    Returns
    -------
    WakeModel
        The deserialized WakeModel object.

    Raises
    ------
    TypeError
        If the type name is not recognized or cannot be loaded.
    """
    from pyvl.wake_models import WakeModelLineExplicitUnsteady

    type_name = group.get_string("type")
    data = group.get_hirearchical_map("data")

    match type_name:
        case _ if custom_types is not None and type_name in custom_types:
            if allow_override:
                cls = custom_types[type_name]
                return cls.load(data)
            raise TypeError(
                f'Type "{type_name}" is registered in custom_types but '
                "allow_override is False."
            )
        case "pyvl.wake_models.WakeModelLineExplicitUnsteady":
            return WakeModelLineExplicitUnsteady.load(data)
        case _:
            if custom_types is not None:
                cls = custom_types.get(type_name)
                if cls is not None:
                    return cls.load(data)
            raise TypeError(
                f'Unknown WakeModel type "{type_name}". '
                "Provide custom_types mapping or check type name."
            )
