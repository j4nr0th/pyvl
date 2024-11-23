"""Config file for NOX."""

import nox


@nox.session
def run_tests(session: nox.Session):
    """Run tests using pytest."""
    session.install(".")
    session.install("pytest", "pytest-cov")
    session.run("pytest", "tests")


@nox.session
def interrogate(session: nox.Session):
    """Check for docstrings."""
    session.install("interrogate")
    session.run("interrogate")
