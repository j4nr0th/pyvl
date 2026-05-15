.. _pyvl.private_c.transforms:

Coordinate Transformations
==========================

The transformation module provides utilities for composing and inverting
coordinate transformations. These are used extensively in reference frame
computations.


Transformation Structure
------------------------

.. c:type:: transformation_t

Represents a combined rotation and translation.

.. code-block:: c

    typedef struct {
        real3_t angles;  // Rotation angles around x, y, and z axis
        real3_t offset;  // Translation offset vector
    } transformation_t;

The rotation is expressed as Euler angles in XYZ convention.


Transformation Composition
--------------------------

.. c:function:: void merge_transformations(const real3x3_t trans_a, const real3_t off_a, const real3x3_t trans_b, const real3_t off_b, real3x3_t *p_trans_out, real3_t *p_off_out)

   Compose two transformations: C(x) = A(B(x))

   Given:
   - A(x) = T_A @ x + r_A
   - B(x) = T_B @ x + r_B

   Computes equivalent transformation C(x) = T_C @ x + r_C where:
   - T_C = T_A @ T_B
   - r_C = r_A + T_A @ r_B

   This is useful when the composed transformation will be applied repeatedly,
   as it reduces the computational cost to that of a single transformation.

   :param trans_a: Transformation matrix of first transformation (A)
   :param off_a: Offset of first transformation (r_A)
   :param trans_b: Transformation matrix of second transformation (B)
   :param off_b: Offset of second transformation (r_B)
   :param p_trans_out: Pointer to receive output transformation matrix
   :param p_off_out: Pointer to receive output offset


Inverse Transformation Composition
----------------------------------

.. c:function:: void merge_transformations_reverse(const real3x3_t trans_a, const real3_t off_a, const real3x3_t trans_b, const real3_t off_b, real3x3_t *p_trans_out, real3_t *p_off_out)

   Compose two inverse transformations: C⁻¹(x) = B⁻¹(A⁻¹(x))

   Given:
   - A⁻¹(x) = T_A^T @ (x - r_A)
   - B⁻¹(x) = T_B^T @ (x - r_B)

   Computes equivalent inverse transformation C⁻¹(x) = T_C^T @ (x - r_C) where:
   - T_C = T_B @ T_A
   - r_C = T_A^T @ (r_A + T_B^T @ r_B)

   :param trans_a: Inverse transformation matrix of A
   :param off_a: Offset of A
   :param trans_b: Inverse transformation matrix of B
   :param off_b: Offset of B
   :param p_trans_out: Pointer to receive output inverse transformation matrix
   :param p_off_out: Pointer to receive output offset