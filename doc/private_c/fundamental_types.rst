.. _pyvl.private_c.fundamental_types:

Fundamental Types
=================

This section documents the core data types and mathematical utilities used throughout
the C implementation.


Core Scalar Type
----------------

.. c:autotype:: real_t
   :file: common.h


3D Vector Type
--------------

.. c:type:: real3_t

   A union type representing a 3D vector with three components. Provides multiple ways
   to access the components: ``x``/``y``/``z``, ``v0``/``v1``/``v2``, or ``data[3]``.


3D Matrix Type
--------------

.. c:type:: real3x3_t

   A union type representing a 3x3 matrix. Supports row access (``row0``/``row1``/``row2``),
   individual element access (``m00``--``m22``), or flat array access (``data[9]``).


Geometric ID Type
-----------------

.. c:type:: geo_id_t

   A structure representing an ID with orientation for mesh elements.

**Special Constants:**

- ``INVALID_ID`` - Sentinel value indicating no valid ID
- ``REVERSED`` - Flag OR-ed with an ID to indicate reverse orientation


Vector Operations
-----------------

The following inline functions perform vector algebra:

.. c:function:: real3_t real3_add(const real3_t a, const real3_t b)

   Add two vectors component-wise.

   :param a: First vector
   :param b: Second vector
   :return: Sum vector

.. c:function:: real3_t real3_sub(const real3_t a, const real3_t b)

   Subtract vector b from vector a.

   :param a: First vector
   :param b: Vector to subtract
   :return: Difference vector

.. c:function:: real_t real3_dot(const real3_t a, const real3_t b)

   Compute the dot product of two vectors.

   :param a: First vector
   :param b: Second vector
   :return: Scalar dot product

.. c:function:: real3_t real3_cross(const real3_t a, const real3_t b)

   Compute the cross product of two vectors (right-hand rule convention).

   :param a: First vector
   :param b: Second vector
   :return: Cross product vector

.. c:function:: real_t real3_mag(const real3_t a)

   Compute the magnitude (length) of a vector.

   :param a: Vector to measure
   :return: Magnitude

.. c:function:: real3_t real3_unit(const real3_t a)

   Return the unit vector (normalized to length 1).

   :param a: Vector to normalize
   :return: Unit vector

.. c:function:: real3_t real3_mul1(const real3_t a, const real_t k)

   Multiply vector by scalar.

   :param a: Vector to scale
   :param k: Scalar multiplier
   :return: Scaled vector

.. c:function:: real3_t real3_neg(const real3_t a)

   Negate vector components.

   :param a: Vector to negate
   :return: Negated vector

.. c:function:: real_t real3_max(const real3_t a)

   Return the maximum component value.

   :param a: Vector to examine
   :return: Maximum component

.. c:function:: bool real3_all_zero(const real3_t a)

   Check whether all components of a vector are exactly zero.

   :param a: Vector to test.
   :return: true if x == 0 && y == 0 && z == 0.


Matrix Operations
-----------------

.. c:function:: real3_t real3x3_vecmul(const real3x3_t a, const real3_t b)

   Multiply matrix by vector.

   :param a: Matrix
   :param b: Vector
   :return: Result vector

.. c:function:: real3x3_t real3x3_matmul(const real3x3_t a, const real3x3_t b)

   Multiply two matrices.

   :param a: Left matrix
   :param b: Right matrix
   :return: Result matrix

.. c:function:: real3_t real3x3_vecmul_transpose(const real3x3_t a, const real3_t b)

   Multiply the transpose of a matrix by a vector (equivalent to treating
   each row of *a* as a vector and computing its dot product with *b*).

   :param a: Matrix.
   :param b: Vector.
   :return: Result vector.

.. c:function:: real3x3_t real3x3_from_angles(const real3_t angles)

   Create rotation matrix from Euler angles (XYZ convention).

   :param angles: Euler angles (x, y, z rotations)
   :return: Rotation matrix

.. c:function:: real3x3_t real3x3_inverse_from_angles(const real3_t angles)

   Create inverse rotation matrix from Euler angles.

   :param angles: Euler angles (x, y, z rotations)
   :return: Inverse rotation matrix

.. c:function:: real3_t angles_from_real3x3(const real3x3_t a)

   Extract Euler angles from rotation matrix.

   :param a: Rotation matrix
   :return: Euler angles

.. c:function:: real_t clamp_angle_to_range(const real_t a)

   Normalize angle to range [0, 2*PI].

   :param a: Angle in radians
   :return: Normalized angle


Geometric ID Utilities
----------------------

.. c:function:: bool id_valid(const geo_id_t *id)

   Check whether a ``geo_id_t`` holds a valid element index.

   :param id: Pointer to the geometric ID.
   :return: true if the value field is not ``INVALID_ID``.

.. c:function:: int geo_id_compare(geo_id_t id1, geo_id_t id2)

   Compare two geometric IDs for equality, including orientation.

   :param id1: First ID.
   :param id2: Second ID.
   :return: 1 if equal (same value and orientation), 0 if different value, -1 if same value but opposite orientation.

.. c:macro:: INVALID_ID

   Sentinel value that should not correspond to any valid element index.
   IDs with this value in their `value` field are considered invalid.

.. c:macro:: REVERSED

   Flag that, when OR-ed with a line or surface ID, indicates reverse
   orientation (the element is traversed opposite to its canonical direction).

Special Constants
~~~~~~~~~~~~~~~~~

   - ``INVALID_ID`` — sentinel for no valid element index
   - ``REVERSED`` — orientation reversal flag


Utility Macros
--------------

.. c:macro:: CVL_PREFETCH(addr, rw, locality)

   Prefetch a memory address into cache. Wrapper around
   ``__builtin_prefetch`` on GCC/clang; no-op on other compilers.

   :param addr: Address to prefetch.
   :param rw: 0 for read, 1 for write.
   :param locality: Locality hint (0-3, higher = more persistent).


Memory Allocator
----------------

.. c:autotype:: allocator_t
   :file: common.h

The allocator provides stateful memory management with allocate, deallocate, and
reallocate callbacks.
