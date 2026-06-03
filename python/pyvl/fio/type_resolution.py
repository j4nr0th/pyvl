"""Type resolution functions using match-case for serialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pyvl._typing import CallableDeserializer
from pyvl.fio.io_common import HirearchicalMap

if TYPE_CHECKING:
    from pyvl.cvl import ReferenceFrame
    from pyvl.flow_conditions import FlowConditions


def reference_frame_from_serial(
    group: HirearchicalMap,
    deserializer: CallableDeserializer,
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

    parent = None
    if "parent" in group:
        parent_group = group.get_hirearchical_map("parent")
        parent = reference_frame_from_serial(
            parent_group, deserializer, custom_types, allow_override
        )

    return (
        ReferenceFrame.load(group, deserializer, parent)
        if parent
        else ReferenceFrame.load(group, deserializer)
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
