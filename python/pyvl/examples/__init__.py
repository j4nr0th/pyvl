"""Example related data."""

from importlib.resources import files
from pathlib import Path

EXAMPLES = {
    "wing1": "Straight, uniform NACA2412 wing with open tips.",
    "fus1": "Basic fuselage.",
}


def example_file_name(name: str) -> Path:
    """Return the full example file path."""
    f_in = files("pyvl.examples").joinpath(name)
    if not f_in.is_file():
        raise ValueError(f"There is no valid resource with the given name ({f_in}).")
    return Path(str(f_in))
