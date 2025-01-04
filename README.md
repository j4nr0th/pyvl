# Python Vortex Lattice solver (`PyVL`)

The aim of this project is to provide a vortex lattice solver, which can handle all kinds
of open and closed meshes. It aims to be highly performant on a single machine and allow
for easy adaptation to different specific applications.

**TODO**:
- add a cool image
- link wiki/pages

**Table of Contents**:
- [Installation from Source](#installation-from-source)
- [Contributing](#contributing)

## Installation from Source

This section describes how to install the package from source. This includes instructions
on how to get a user version, as well as how to install the package for development.

### Dependencies

The first step is installing the dependencies. In order to be able to build the package,
you will need:

- [`CMake`](https://cmake.org/),
- Your system's C compiler (on Linux it's `gcc` and on Windows it is `msvc`),
- Support for OpenMP 3.0 (`msvc` will use `llvm` to support it),
- Python 3.10 or newer.

### Installing the Package

To install the `PyVL` package in the active Python environment. Assuming that you cloned
the repository to the directory `pyvl`, simply use `pip` like so:
```bash
python -m pip install pyvl
```

This will build the package and install it.

### Development Version

If you instead want to work on the package, you can specify `dev` and/or `docs` as
optional dependencies for the package. You also probably want to then install the package
as editable:
```bash
python -m pip install -e pyvl[dev,docs]
```


## Contributing

This section outlines guidelines on how to contribute to the project. While all of these
steps are not necessary, they makes other peoples' lives easier. These steps are all
assuming you have [installed the development version](#development-version) of the
package.

### Using [`pre-commit`](https://pre-commit.com/)

This package uses [`pre-commit`](https://pre-commit.com/) in order to run
[git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) prior to allowing
you to make a commit. These include Python type checking, C and Python formatting, and
spellchecking.

After you clone the repository, you must then initialize these by running the following
command in your development environment:
```bash
pre-commit install
```

Note that this needs to be done only once pre commit.

### Running Tests

There are three groups of test which are included in the package:

- C tests, which check the low-level mathematical functions (stored in sub-directories
    of `test` directory).
- Python tests, which check the rest of the module, from the C types, to the pure Python
    code (stored in `test/pytests`).
- Python doctests, which are in the docstrings of the Python types/functions. These
    are in docstrings of more important types in the `python/pyvl` directory.

The Python tests can be run automatically by calling `pytest`. To run C tests, you can
use [`CMake`](https://cmake.org/).

### Building the Documentation

Documentation for `pyvl` is written with
[`Sphinx`](https://www.sphinx-doc.org/en/master/). As such, you can build it by going
to the `doc` directory, then executing `make html`, which should work both on Linux and
Windows.

### Using [`nox`](https://nox.thea.codes/en/stable/index.html)

To automatically test building, code tests, and building the documentation in clean
environment, [`nox`](https://nox.thea.codes/en/stable/index.html) can be used. To display
the list of available nox sessions:
```bash
nox --list
```

If you just want to run all the default sessions, you can also just run:
```bash
nox
```
