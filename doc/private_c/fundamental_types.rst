.. _pyvl.private_c.fundamental_types:

Fundamental Types
=================

This section documents the core data types and mathematical utilities used throughout
the C implementation.


Core Scalar Type
----------------

.. c:type:: real_t

Alias for ``double``. Used throughout the codebase to represent real (floating-point) numbers.


3D Vector Type
--------------

.. c:type:: real3_t

A union type representing a 3D vector with three components. Provides multiple ways
to access the components:

- ``x``, ``y``, ``z`` - Named component access
- ``v0``, ``v1``, ``v2`` - Indexed component access
- ``data[3]`` - Array access


3D Matrix Type
--------------

.. c:type:: real3x3_t

A union type representing a 3x3 matrix. Supports:

- ``row0``, ``row1``, ``row2`` - Row access as real3_t
- ``m00`` through ``m22`` - Individual element access
- ``data[9]`` - Flat array access


Geometric ID Type
-----------------

.. c:type:: geo_id_t

A structure representing an ID with orientation for mesh elements.

.. code-block:: c

    typedef struct {
        uint32_t value : 31;
        uint32_t orientation : 1;
    } geo_id_t;

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


Memory Allocator
----------------

.. c:type:: allocator_t

Callback-based memory allocation interface.

.. code-block:: c

    typedef struct {
        void *(*allocate)(void *state, size_t size);
        void (*deallocate)(void *state, void *ptr);
        void *(*reallocate)(void *state, void *ptr, size_t new_size);
        void *state;
    } allocator_t;

The allocator provides stateful memory management with allocate, deallocate, and
reallocate callbacks.
