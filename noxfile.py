"""Config file for NOX."""

import nox


@nox.session
def run_tests(session: nox.Session):
    """Run tests using pytest."""
    session.install(".")
    session.install("pytest", "pytest-cov")
    session.run("pytest", "test/pytests")


@nox.session
def interrogate(session: nox.Session):
    """Check for docstrings."""
    session.install("interrogate")
    session.run("interrogate")


@nox.session
def build_docs(session: nox.Session):
    """Build the project documentation."""
    session.install(".[docs]")
    session.run("sphinx-build", "-M", "html", "doc", "doc/_build")
