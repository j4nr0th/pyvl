.. _pyvl.output_settings:

.. currentmodule:: pyvl

Output Settings
===============

To control how the IO is done, the :class:`OutputSettings` is used. Currently,
the file format can be selected between `JSON <https://www.json.org/>`_, which is
a popular plain-text file format, and `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_,
which is commonly used in scientific computing and is a binary format.

To choose a file name, the :attr:`OutputSettings.naming_callback` is used. This :class:`Callable`
will be called with the iteration number and time at the current time and expect a path/name of the
file where the output should be written to.

.. autoclass:: OutputSettings
    :members:
