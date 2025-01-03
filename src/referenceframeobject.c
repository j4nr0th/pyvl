//
// Created by jan on 29.11.2024.
//
#include "referenceframeobject.h"
#include <numpy/arrayobject.h>
// This must be after the arrayobject
#include "common.h"

static PyObject *pyvl_reference_frame_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    double theta_x = 0, theta_y = 0, theta_z = 0;
    double offset_x = 0, offset_y = 0, offset_z = 0;
    PyVL_ReferenceFrame *parent;
    PyObject *p = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|(ddd)(ddd)O", (char *[4]){"offset", "theta", "parent", NULL},
                                     &offset_x, &offset_y, &offset_z, &theta_x, &theta_y, &theta_z, &p))
    {
        return NULL;
    }
    if (p == NULL || Py_IsNone(p))
    {
        parent = NULL;
    }
    else if (!PyObject_TypeCheck(p, &pyvl_reference_frame_type))
    {
        PyErr_Format(PyExc_TypeError, "Argument \"parent\" must be a ReferenceFrame object, but it was %R", Py_TYPE(p));
        return NULL;
    }
    else
    {
        parent = (PyVL_ReferenceFrame *)p;
    }
    theta_x = clamp_angle_to_range(theta_x);
    theta_y = clamp_angle_to_range(theta_y);
    theta_z = clamp_angle_to_range(theta_z);

    PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)type->tp_alloc(type, 0);
    if (!this)
    {
        return NULL;
    }
    Py_XINCREF(parent);

    this->parent = parent;
    this->transformation = (transformation_t){
        .angles = {.x = theta_x, .y = theta_y, .z = theta_z},
        .offset = {.x = offset_x, .y = offset_y, .z = offset_z},
    };
    return (PyObject *)this;
}

static void pyvl_reference_frame_dealloc(PyObject *self)
{
    PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    Py_XDECREF(this->parent);
    Py_TYPE(this)->tp_free(this);
}

static PyObject *pyvl_reference_frame_repr(PyObject *self)
{
    // Recursion should never happen, since there's no real way to make a cycle due to being unable to set
    // the parent in Python code.

    // const int repr_res = Py_ReprEnter(self);
    // if (repr_res < 0) return NULL;
    // if (repr_res > 0) return PyUnicode_FromString("...");

    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    PyObject *out;
    unsigned len = snprintf(NULL, 0, "(%g, %g, %g), (%g, %g, %g)", this->transformation.angles.x,
                            this->transformation.angles.y, this->transformation.angles.z, this->transformation.offset.x,
                            this->transformation.offset.y, this->transformation.offset.z);
    char *buffer = PyMem_Malloc((len + 1) * sizeof *buffer);
    if (!buffer)
        return NULL;
    (void)snprintf(buffer, (len + 1) * sizeof(*buffer), "(%g, %g, %g), (%g, %g, %g)", this->transformation.angles.x,
                   this->transformation.angles.y, this->transformation.angles.z, this->transformation.offset.x,
                   this->transformation.offset.y, this->transformation.offset.z);

    if (this->parent)
    {
        out = PyUnicode_FromFormat("ReferenceFrame(%s, parent=%R)", buffer, (PyObject *)this->parent);
    }
    else
    {
        out = PyUnicode_FromFormat("ReferenceFrame(%s)", buffer);
    }
    PyMem_Free(buffer);
    // Py_ReprLeave(self);
    return out;
}

static PyObject *pyvl_reference_frame_get_parent(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    if (!this->parent)
    {
        Py_RETURN_NONE;
    }
    Py_INCREF(this->parent);
    return (PyObject *)this->parent;
}

static PyObject *pyvl_reference_frame_get_offset(PyObject *self, void *Py_UNUSED(closure))
{
    static const npy_intp dim = 3;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
    if (!out)
        return NULL;
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    double *const p_out = PyArray_DATA(out);
    if (!p_out)
    {
        // I don't think this would ever even happen.
        Py_DECREF(out);
        return NULL;
    }
    p_out[0] = this->transformation.offset.x;
    p_out[1] = this->transformation.offset.y;
    p_out[2] = this->transformation.offset.z;
    return (PyObject *)out;
}

static PyObject *pyvl_reference_frame_get_angles(PyObject *self, void *Py_UNUSED(closure))
{
    static const npy_intp dim = 3;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
    if (!out)
        return NULL;
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    double *const p_out = PyArray_DATA(out);
    if (!p_out)
    {
        // I don't think this would ever even happen.
        Py_DECREF(out);
        return NULL;
    }
    p_out[0] = this->transformation.angles.x;
    p_out[1] = this->transformation.angles.y;
    p_out[2] = this->transformation.angles.z;
    return (PyObject *)out;
}

static PyObject *pyvl_reference_frame_get_rotation_matrix(PyObject *self, void *Py_UNUSED(closure))
{
    static const npy_intp dims[2] = {3, 3};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
        return NULL;
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    double *const p_out = PyArray_DATA(out);
    if (!p_out)
    {
        // I don't think this would ever even happen.
        Py_DECREF(out);
        return NULL;
    }
    const real3x3_t mat = real3x3_from_angles(this->transformation.angles);
    for (unsigned i = 0; i < 9; ++i)
    {
        p_out[i] = mat.data[i];
    }
    return (PyObject *)out;
}

static PyObject *pyvl_reference_frame_get_rotation_matrix_inverse(PyObject *self, void *Py_UNUSED(closure))
{
    static const npy_intp dims[2] = {3, 3};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
        return NULL;
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    double *const p_out = PyArray_DATA(out);
    if (!p_out)
    {
        // I don't think this would ever even happen.
        Py_DECREF(out);
        return NULL;
    }
    const real3x3_t mat = real3x3_inverse_from_angles(this->transformation.angles);
    for (unsigned i = 0; i < 9; ++i)
    {
        p_out[i] = mat.data[i];
    }
    return (PyObject *)out;
}

static PyObject *pyvl_reference_frame_get_parents(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    unsigned parent_cnt = 0;
    for (PyVL_ReferenceFrame *p = this->parent; p; p = p->parent)
        parent_cnt += 1;
    PyObject *out = PyTuple_New(parent_cnt);
    unsigned i = 0;
    for (PyVL_ReferenceFrame *p = this->parent; p; p = p->parent)
    {
        Py_INCREF(p);
        PyTuple_SET_ITEM(out, i, (PyObject *)p);
        i += 1;
    }
    return out;
}

static PyObject *pyvl_reference_frame_rich_compare(PyObject *self, PyObject *other, const int op)
{
    static const real_t tol = 1e-10;
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    if (Py_TYPE(other) != &pyvl_reference_frame_type)
    {
        Py_RETURN_FALSE;
    }
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    const PyVL_ReferenceFrame *that = (PyVL_ReferenceFrame *)other;

    bool result = true;
    while (this && that)
    {
        const real3_t dr1 = real3_sub(this->transformation.offset, that->transformation.offset);
        if (real3_dot(dr1, dr1) < tol)
        {
            result = false;
            break;
        }
        const real3_t dr2 = real3_sub(this->transformation.angles, that->transformation.angles);
        if (real3_dot(dr2, dr2) < tol)
        {
            result = false;
            break;
        }
        this = this->parent;
        that = that->parent;
    }
    if (this || that)
    {
        result = false;
    }

    result = (op == Py_EQ) ? result : !result;
    if (!result)
    {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyGetSetDef pyvl_reference_frame_getset[] = {
    {.name = "parent",
     .get = pyvl_reference_frame_get_parent,
     .set = NULL,
     .doc = "ReferenceFrame | None : Return what reference frame the current one is defined relative to.\n",
     .closure = NULL},
    {.name = "offset",
     .get = pyvl_reference_frame_get_offset,
     .set = NULL,
     .doc = "array : Vector determining the offset of the reference frame in relative to its parent.",
     .closure = NULL},
    {.name = "angles",
     .get = pyvl_reference_frame_get_angles,
     .set = NULL,
     .doc = "array : Vector determining the rotation of the reference frame in parent's frame.",
     .closure = NULL},
    {.name = "rotation_matrix",
     .get = pyvl_reference_frame_get_rotation_matrix,
     .set = NULL,
     .doc = "array : Matrix representing rotation of the reference frame.\n",
     .closure = NULL},
    {.name = "rotation_matrix_inverse",
     .get = pyvl_reference_frame_get_rotation_matrix_inverse,
     .set = NULL,
     .doc = "array : Matrix representing inverse rotation of the reference frame.",
     .closure = NULL},
    {.name = "parents",
     .get = pyvl_reference_frame_get_parents,
     .set = NULL,
     .doc = "tuple[ReferenceFrame, ...] : Tuple of all parents of this reference frame.\n",
     .closure = NULL},
    {0},
};

static bool prepare_for_transformation(Py_ssize_t nargs, PyObject *const CVL_ARRAY_ARG(args, static nargs),
                                       PyArrayObject **p_in, PyArrayObject **p_out, npy_intp *p_dim,
                                       const npy_intp **p_dims)
{
    if (nargs != 1 && nargs != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Method only takes 1 or two parameters");
        return false;
    }
    if (nargs == 2 && !PyArray_Check(args[1]))
    {
        PyErr_SetString(PyExc_TypeError, "Second argument must be an array.");
        return false;
    }
    PyArrayObject *const in_array = (PyArrayObject *)PyArray_FromAny(args[0], PyArray_DescrFromType(NPY_FLOAT64), 1, 0,
                                                                     NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!in_array)
    {
        return false;
    }
    const npy_intp dim_in = PyArray_NDIM(in_array);
    const npy_intp *dims_in = PyArray_DIMS(in_array);
    if (dims_in[dim_in - 1] != 3)
    {
        PyErr_Format(PyExc_ValueError,
                     "Input array does not have the last axis with 3 dimensions "
                     "(shape is (..., %u) instead of (..., 3)).",
                     (unsigned)dims_in[dim_in - 1]);
        Py_DECREF(in_array);
        return false;
    }
    PyArrayObject *out_array;
    if (nargs == 2)
    {
        out_array = (PyArrayObject *)args[1];
        if (!(PyArray_FLAGS(out_array) & NPY_ARRAY_C_CONTIGUOUS))
        {
            PyErr_SetString(PyExc_ValueError, "Output array is not C-contiguous.");
            Py_DECREF(in_array);
            return false;
        }
        if (PyArray_TYPE(out_array) != NPY_DOUBLE)
        {
            PyErr_SetString(PyExc_ValueError, "Output array is not of C double type.");
            Py_DECREF(in_array);
            return false;
        }

        const npy_intp dim_out = PyArray_NDIM(out_array);
        const npy_intp *dims_out = PyArray_DIMS(out_array);
        if (dim_out != dim_in)
        {
            PyErr_Format(PyExc_ValueError,
                         "Output array does not have the same number of axis as the input "
                         "array (in: %u, out: %u).",
                         (unsigned)dim_in, (unsigned)dim_out);
            Py_DECREF(in_array);
            return false;
        }
        if (memcmp(dims_in, dims_out, sizeof(*dims_in) * dim_in) != 0)
        {
            PyErr_SetString(PyExc_ValueError, "Output array does not have the same exact shape as the input array.");
            Py_DECREF(in_array);
            return false;
        }
        Py_INCREF(out_array);
    }
    else
    {
        out_array = (PyArrayObject *)PyArray_NewLikeArray(in_array, NPY_CORDER, NULL, 0);
        if (!out_array)
        {
            Py_DECREF(in_array);
            return false;
        }
    }
    *p_in = in_array;
    *p_out = out_array;
    *p_dim = dim_in;
    *p_dims = dims_in;
    return true;
}

static PyObject *pyvl_reference_frame_from_parent_with_offset(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return NULL;
    }

    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    const real3x3_t mat = real3x3_from_angles(this->transformation.angles);
    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3_add(real3x3_vecmul(mat, p_in[i]), this->transformation.offset);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_from_parent_without_offset(PyObject *self, PyObject *const *args,
                                                                 Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return NULL;
    }

    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    const real3x3_t mat = real3x3_from_angles(this->transformation.angles);
    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3x3_vecmul(mat, p_in[i]);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_to_parent_with_offset(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return NULL;
    }

    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    const real3x3_t mat = real3x3_inverse_from_angles(this->transformation.angles);
    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3x3_vecmul(mat, real3_sub(p_in[i], this->transformation.offset));
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_to_parent_without_offset(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return NULL;
    }

    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    const real3x3_t mat = real3x3_inverse_from_angles(this->transformation.angles);
    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3x3_vecmul(mat, p_in[i]);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_from_global_with_offset(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return NULL;
    }

    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    real3x3_t mat = real3x3_from_angles(this->transformation.angles);
    real3_t off = this->transformation.offset;
    for (const PyVL_ReferenceFrame *p = this; p->parent; p = p->parent)
    {
        merge_transformations(real3x3_from_angles(p->transformation.angles), p->transformation.offset, mat, off, &mat,
                              &off);
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3_add(real3x3_vecmul(mat, p_in[i]), off);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_from_global_without_offset(PyObject *self, PyObject *const *args,
                                                                 Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return NULL;
    }

    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    real3x3_t mat = real3x3_from_angles(this->transformation.angles);
    for (const PyVL_ReferenceFrame *p = this; p->parent; p = p->parent)
    {
        mat = real3x3_matmul(real3x3_from_angles(p->transformation.angles), mat);
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3x3_vecmul(mat, p_in[i]);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_to_global_with_offset(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return NULL;
    }

    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    real3x3_t mat = real3x3_inverse_from_angles(this->transformation.angles);
    real3_t off = this->transformation.offset;
    for (const PyVL_ReferenceFrame *p = this; p->parent; p = p->parent)
    {
        merge_transformations_reverse(real3x3_inverse_from_angles(p->transformation.angles), p->transformation.offset,
                                      mat, off, &mat, &off);
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3_sub(real3x3_vecmul(mat, p_in[i]), off);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_to_global_without_offset(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return NULL;
    }

    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    real3x3_t mat = real3x3_inverse_from_angles(this->transformation.angles);
    for (const PyVL_ReferenceFrame *p = this; p->parent; p = p->parent)
    {
        mat = real3x3_matmul(mat, real3x3_inverse_from_angles(p->transformation.angles));
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3x3_vecmul(mat, p_in[i]);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_rotate_x(PyObject *self, PyObject *arg)
{
    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    const double theta = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
        return NULL;

    PyVL_ReferenceFrame *const new =
        (PyVL_ReferenceFrame *)pyvl_reference_frame_type.tp_alloc(&pyvl_reference_frame_type, 0);
    if (!new)
        return NULL;
    new->transformation = this->transformation;
    new->parent = this->parent;
    Py_XINCREF(this->parent);
    new->transformation.angles.x = clamp_angle_to_range(new->transformation.angles.x + theta);
    return (PyObject *)new;
}

static PyObject *pyvl_reference_frame_rotate_y(PyObject *self, PyObject *arg)
{
    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    const double theta = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
        return NULL;

    PyVL_ReferenceFrame *const new =
        (PyVL_ReferenceFrame *)pyvl_reference_frame_type.tp_alloc(&pyvl_reference_frame_type, 0);
    if (!new)
        return NULL;
    new->transformation = this->transformation;
    new->parent = this->parent;
    Py_XINCREF(this->parent);
    new->transformation.angles.y = clamp_angle_to_range(new->transformation.angles.y + theta);
    return (PyObject *)new;
}

static PyObject *pyvl_reference_frame_rotate_z(PyObject *self, PyObject *arg)
{
    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    const double theta = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
        return NULL;

    PyVL_ReferenceFrame *const new =
        (PyVL_ReferenceFrame *)pyvl_reference_frame_type.tp_alloc(&pyvl_reference_frame_type, 0);
    if (!new)
        return NULL;
    new->transformation = this->transformation;
    new->parent = this->parent;
    Py_XINCREF(this->parent);
    new->transformation.angles.z = clamp_angle_to_range(new->transformation.angles.z + theta);
    return (PyObject *)new;
}

static PyObject *pyvl_reference_frame_with_offset(PyObject *self, PyObject *arg)
{
    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    PyArrayObject *const off_array =
        (PyArrayObject *)PyArray_FromAny(arg, PyArray_DescrFromType(NPY_FLOAT64), 1, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!off_array)
        return NULL;
    const npy_intp *p_n = PyArray_DIMS(off_array);
    if (*p_n != 3)
    {
        PyErr_Format(PyExc_ValueError, "Input array must have 3 element, instead %u were given.", (unsigned)*p_n);
        Py_DECREF(off_array);
        return NULL;
    }
    const npy_float64 *p_in = PyArray_DATA(off_array);
    const real3_t new_offset = {.v0 = p_in[0], .v1 = p_in[1], .v2 = p_in[2]};
    Py_DECREF(off_array);

    PyVL_ReferenceFrame *const new =
        (PyVL_ReferenceFrame *)pyvl_reference_frame_type.tp_alloc(&pyvl_reference_frame_type, 0);
    if (!new)
        return NULL;
    new->transformation = this->transformation;
    new->parent = this->parent;
    Py_XINCREF(this->parent);
    new->transformation.offset = new_offset;
    return (PyObject *)new;
}

static PyObject *pyvl_reference_frame_at_time(PyObject *self, PyObject *arg)
{
    const double time = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
        return NULL;
    (void)time;
    PyVL_ReferenceFrame *const new =
        (PyVL_ReferenceFrame *)pyvl_reference_frame_type.tp_alloc(&pyvl_reference_frame_type, 0);
    if (!new)
        return NULL;
    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    new->transformation = this->transformation;
    if (this->parent)
    {
        //  Evolve the parent.
        PyVL_ReferenceFrame *new_parent =
            (PyVL_ReferenceFrame *)PyObject_CallMethod((PyObject *)this->parent, "at_time", "O", arg);
        if (!new_parent)
        {
            Py_DECREF(new);
            return NULL;
        }
        new->parent = new_parent;
    }
    else
    {
        new->parent = NULL;
    }
    return (PyObject *)new;
}

static PyObject *pyvl_matrix_to_angles(PyObject *Py_UNUSED(module), PyObject *arg)
{
    PyArrayObject *const array = (PyArrayObject *)PyArray_FROMANY(arg, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!array)
        return NULL;
    const npy_intp *const dims = PyArray_DIMS(array);
    if (dims[0] != 3 || dims[1] != 3)
    {
        PyErr_Format(PyExc_ValueError, "Array was not a (3, 3) array, instead it was (%u, %u).", (unsigned)dims[0],
                     (unsigned)dims[1]);
        return NULL;
    }
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, (const npy_intp[1]){3}, NPY_DOUBLE);
    if (!out)
    {
        Py_DECREF(array);
        return NULL;
    }
    const double *restrict mat = PyArray_DATA(array);
    double *const restrict angles = PyArray_DATA(out);

    const real3x3_t matrix = {
        .m00 = mat[0],
        .m01 = mat[1],
        .m02 = mat[2],
        .m10 = mat[3],
        .m11 = mat[4],
        .m12 = mat[5],
        .m20 = mat[6],
        .m21 = mat[7],
        .m22 = mat[8],
    };
    const real3_t a = angles_from_real3x3(matrix);
    angles[0] = a.x;
    angles[1] = a.y;
    angles[2] = a.z;

    Py_DECREF(array);
    return (PyObject *)out;
}

static PyObject *pyvl_reference_frame_save(PyObject *self, PyObject *arg)
{
    if (!PyMapping_Check(arg))
    {
        PyErr_Format(PyExc_TypeError, "The input parameter is not a mapping.");
        return NULL;
    }
    PyObject *const off_array = pyvl_reference_frame_get_offset(self, NULL);
    PyObject *const rot_array = pyvl_reference_frame_get_angles(self, NULL);
    if (!off_array || !rot_array)
    {
        Py_XDECREF(off_array);
        Py_XDECREF(rot_array);
        return NULL;
    }
    // if (this->parent)
    // {
    //     PyObject *const out_group = PyObject_CallMethod(arg, "create_group", "s", "parent");
    //     if (!out_group)
    //     {
    //         Py_DECREF(off_array);
    //         Py_DECREF(rot_array);
    //         return NULL;
    //     }
    //     PyObject *const res = pyvl_reference_frame_save((PyObject *)this->parent, out_group);
    //     Py_DECREF(out_group);
    //     Py_XDECREF(res);
    //     if (!res)
    //         return NULL;
    // }
    const int res1 = PyMapping_SetItemString(arg, "offset", off_array);
    Py_DECREF(off_array);
    const int res2 = PyMapping_SetItemString(arg, "angles", rot_array);
    Py_DECREF(rot_array);
    if (res1 < 0 || res2 < 0)
        return NULL;
    PyObject *type_name = PyUnicode_FromString(pyvl_reference_frame_type.tp_name);
    if (!type_name)
        return NULL;
    const int res3 = PyMapping_SetItemString(arg, "type", type_name);
    Py_DECREF(type_name);
    if (res3 < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject *pyvl_reference_frame_load(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *group, *parent = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char *[3]){"group", "parent", NULL}, &group, &parent))
    {
        return NULL;
    }
    if (parent)
    {
        if (Py_IsNone(parent))
        {
            parent = NULL;
        }
        else if (!PyObject_TypeCheck(parent, &pyvl_reference_frame_type))
        {
            PyErr_Format(PyExc_TypeError, "Parent was neither None nor a ReferenceFrame (instead it was %R).",
                         Py_TYPE(parent));
            return NULL;
        }
    }
    if (!PyMapping_Check(group))
    {
        PyErr_Format(PyExc_TypeError, "The input parameter is not a mapping.");
        return NULL;
    }
    // PyObject *const type_name = PyMapping_GetItemString(group, "type");
    // if (!type_name)
    //     return NULL;
    // Py_ssize_t len;
    // const char *const full_name = PyUnicode_AsUTF8AndSize(type_name, &len);
    // if (!full_name)
    // {
    //     Py_DECREF(type_name);
    //     return NULL;
    // }
    // const char *const split = strrchr(full_name, '.');
    // if (!split)
    // {
    //     Py_DECREF(type_name);
    //     PyErr_Format(PyExc_ValueError, "The type name did not contain any \".\" characters.");
    //     return NULL;
    // }
    // const unsigned mod_len = split - full_name;
    // char *const buffer = PyMem_Malloc(sizeof(*buffer) * (mod_len));
    // if (!buffer)
    // {
    //     Py_DECREF(type_name);
    //     return NULL;
    // }
    // memcpy(buffer, full_name, mod_len - 1);
    // buffer[mod_len - 1] = 0;
    // PyObject *module = PyImport_ImportModule(buffer);
    // PyMem_Free(buffer);
    // if (!module)
    // {
    //     Py_DECREF(type_name);
    //     return NULL;
    // }
    PyTypeObject *type = (PyTypeObject *)self; // PyObject_GetAttrString(module, split + 1);
    // Py_DECREF(type_name);
    // if (!type)
    //     return NULL;

    PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)type->tp_alloc(type, 0);
    // Py_DECREF(type);
    if (!this)
        return NULL;
    PyObject *const off_val = PyMapping_GetItemString(group, "offset");
    PyObject *const rot_val = PyMapping_GetItemString(group, "angles");
    if (!off_val || !rot_val)
    {
        Py_XDECREF(off_val);
        Py_XDECREF(rot_val);
        return NULL;
    }
    PyArrayObject *const off_array = (PyArrayObject *)PyArray_FromAny(off_val, PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
                                                                      NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    Py_DECREF(off_val);
    PyArrayObject *const rot_array = (PyArrayObject *)PyArray_FromAny(rot_val, PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
                                                                      NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    Py_DECREF(rot_val);
    if (!off_array || !rot_array)
    {
        Py_XDECREF(off_array);
        Py_XDECREF(rot_array);
        Py_DECREF(this);
        return NULL;
    }
    if (PyArray_SIZE(off_array) != 3)
    {
        PyErr_Format(PyExc_ValueError, "Offset array did not have 3 elements, but had %u instead.",
                     (unsigned)PyArray_SIZE(off_array));
        Py_DECREF(off_array);
        Py_DECREF(rot_array);
        Py_DECREF(this);
        return NULL;
    }
    if (PyArray_SIZE(rot_array) != 3)
    {
        PyErr_Format(PyExc_ValueError, "Angle array did not have 3 elements, but had %u instead.",
                     (unsigned)PyArray_SIZE(rot_array));
        Py_DECREF(off_array);
        Py_DECREF(rot_array);
        Py_DECREF(this);
        return NULL;
    }

    const npy_float64 *const offset_ptr = PyArray_DATA(off_array);
    const npy_float64 *const angles_ptr = PyArray_DATA(rot_array);

    this->transformation.offset = (real3_t){.x = offset_ptr[0], .y = offset_ptr[1], .z = offset_ptr[2]};
    this->transformation.angles = (real3_t){.x = angles_ptr[0], .y = angles_ptr[1], .z = angles_ptr[2]};

    Py_DECREF(off_array);
    Py_DECREF(rot_array);
    this->parent = (PyVL_ReferenceFrame *)parent;
    Py_XINCREF(parent);

    return (PyObject *)this;
}

static PyObject *pyvl_reference_frame_add_velocity(PyObject *Py_UNUSED(self), PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Function takes 2 parameters, but was called with %u\n", (unsigned)nargs);
        return NULL;
    }
    const PyArrayObject *const in_array =
        pyvl_ensure_array(args[0], 2, (const npy_intp[2]){0, 3}, 0, NPY_FLOAT64, "Position array");
    const npy_intp required_dims[2] = {PyArray_DIM(in_array, 0), 3};
    const PyArrayObject *const out_array =
        pyvl_ensure_array(args[1], 2, required_dims, NPY_ARRAY_WRITEABLE, NPY_FLOAT64, "Velocity array");
    if (!out_array)
        return NULL;

    Py_RETURN_NONE;
}

static PyMethodDef pyvl_reference_frame_methods[] = {
    {.ml_name = "from_parent_with_offset",
     .ml_meth = (void *)pyvl_reference_frame_from_parent_with_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "from_parent_with_offset(x: array, out: out_array | None = None) -> out_array\n"
               "Map position vector from parent reference frame to the child reference frame.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "x : (N, 3) array\n"
               "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in parent reference frame.\n"
               "out : (N, 3) array, optional"
               "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
               "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
               "    C-contiguous, and writable.\n"
               "Returns\n"
               "-------\n"
               "(N, 3) array\n"
               "    Position vectors mapped to the child reference frame. If the ``out`` parameter was\n"
               "    specified, this return value will be the same object. If ``out`` was not specified,\n"
               "    then a new array will be allocated."},
    {.ml_name = "from_parent_without_offset",
     .ml_meth = (void *)pyvl_reference_frame_from_parent_without_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "from_parent_without_offset(x: array, out: out_array | None = None) -> out_array\n"
               "Map direction vector from parent reference frame to the child reference frame.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "x : (N, 3) array\n"
               "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in parent reference frame.\n"
               "out : (N, 3) array, optional\n"
               "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
               "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
               "    C-contiguous, and writable.\n"
               "Returns\n"
               "-------\n"
               "(N, 3) array\n"
               "    Direction vectors mapped to the child reference frame. If the ``out`` parameter was\n"
               "    specified, this return value will be the same object. If ``out`` was not specified,\n"
               "    then a new array will be allocated."},
    {.ml_name = "to_parent_with_offset",
     .ml_meth = (void *)pyvl_reference_frame_to_parent_with_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "to_parent_with_offset(x: array, out: out_array | None = None) -> out_array\n"
               "Map position vector from child reference frame to the parent reference frame.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "x : (N, 3) array\n"
               "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in child reference frame.\n"
               "out : (N, 3) array, optional"
               "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
               "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
               "    C-contiguous, and writable.\n"
               "Returns\n"
               "-------\n"
               "(N, 3) array\n"
               "    Position vectors mapped to the parent reference frame. If the ``out`` parameter was\n"
               "    specified, this return value will be the same object. If ``out`` was not specified,\n"
               "    then a new array will be allocated."},
    {.ml_name = "to_parent_without_offset",
     .ml_meth = (void *)pyvl_reference_frame_to_parent_without_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "to_parent_without_offset(x: array, out: out_array | None = None) -> out_array\n"
               "Map direction vector from child reference frame to the parent reference frame.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "x : (N, 3) array\n"
               "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in child reference frame.\n"
               "out : (N, 3) array, optional\n"
               "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
               "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
               "    C-contiguous, and writable.\n"
               "Returns\n"
               "-------\n"
               "(N, 3) array\n"
               "    Direction vectors mapped to the parent reference frame. If the ``out`` parameter was\n"
               "    specified, this return value will be the same object. If ``out`` was not specified,\n"
               "    then a new array will be allocated."},
    {.ml_name = "from_global_with_offset",
     .ml_meth = (void *)pyvl_reference_frame_from_global_with_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "from_global_with_offset(x: array, out: out_array | None = None) -> out_array\n"
               "Map position vector from global reference frame to the child reference frame.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "x : (N, 3) array\n"
               "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in global reference frame.\n"
               "out : (N, 3) array, optional"
               "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
               "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
               "    C-contiguous, and writable.\n"
               "Returns\n"
               "-------\n"
               "(N, 3) array\n"
               "    Position vectors mapped to the child reference frame. If the ``out`` parameter was\n"
               "    specified, this return value will be the same object. If ``out`` was not specified,\n"
               "    then a new array will be allocated."},
    {.ml_name = "from_global_without_offset",
     .ml_meth = (void *)pyvl_reference_frame_from_global_without_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "from_global_without_offset(x: array, out: out_array | None = None) -> out_array\n"
               "Map direction vector from global reference frame to the child reference frame.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "x : (N, 3) array\n"
               "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in global reference frame.\n"
               "out : (N, 3) array, optional\n"
               "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
               "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
               "    C-contiguous, and writable.\n"
               "Returns\n"
               "-------\n"
               "(N, 3) array\n"
               "    Direction vectors mapped to the child reference frame. If the ``out`` parameter was\n"
               "    specified, this return value will be the same object. If ``out`` was not specified,\n"
               "    then a new array will be allocated."},
    {.ml_name = "to_global_with_offset",
     .ml_meth = (void *)pyvl_reference_frame_to_global_with_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "to_global_with_offset(x: array, out: out_array | None = None) -> out_array\n"
               "Map position vector from child reference frame to the parent reference frame.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "x : (N, 3) array\n"
               "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in child reference frame.\n"
               "out : (N, 3) array, optional"
               "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
               "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
               "    C-contiguous, and writable.\n"
               "Returns\n"
               "-------\n"
               "(N, 3) array\n"
               "    Position vectors mapped to the global reference frame. If the ``out`` parameter was\n"
               "    specified, this return value will be the same object. If ``out`` was not specified,\n"
               "    then a new array will be allocated."},
    {.ml_name = "to_global_without_offset",
     .ml_meth = (void *)pyvl_reference_frame_to_global_without_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "to_global_without_offset(x: array, out: out_array | None = None) -> out_array\n"
               "Map direction vector from child reference frame to the global reference frame.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "x : (N, 3) array\n"
               "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in child reference frame.\n"
               "out : (N, 3) array, optional\n"
               "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
               "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
               "    C-contiguous, and writable.\n"
               "Returns\n"
               "-------\n"
               "(N, 3) array\n"
               "    Direction vectors mapped to the global reference frame. If the ``out`` parameter was\n"
               "    specified, this return value will be the same object. If ``out`` was not specified,\n"
               "    then a new array will be allocated."},
    {.ml_name = "rotate_x",
     .ml_meth = pyvl_reference_frame_rotate_x,
     .ml_flags = METH_O,
     .ml_doc = "rotate_x(theta_x: float, /) -> Self\n"
               "Create a copy of the frame rotated around the x-axis.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "theta_x : float\n"
               "    Angle by which to rotate the reference frame by.\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    Reference frame rotated around the x-axis by the specified angle.\n"},
    {.ml_name = "rotate_y",
     .ml_meth = pyvl_reference_frame_rotate_y,
     .ml_flags = METH_O,
     .ml_doc = "rotate_y(theta_y: float, /) -> Self\n"
               "Create a copy of the frame rotated around the y-axis.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "theta_y : float\n"
               "    Angle by which to rotate the reference frame by.\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    Reference frame rotated around the y-axis by the specified angle.\n"},
    {.ml_name = "rotate_z",
     .ml_meth = pyvl_reference_frame_rotate_z,
     .ml_flags = METH_O,
     .ml_doc = "rotate_z(theta_z: float, /) -> Self\n"
               "Create a copy of the frame rotated around the z-axis.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "theta_z : float\n"
               "    Angle by which to rotate the reference frame by.\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    Reference frame rotated around the z-axis by the specified angle.\n"},
    {.ml_name = "with_offset",
     .ml_meth = pyvl_reference_frame_with_offset,
     .ml_flags = METH_O,
     .ml_doc = "with_offset(offset: VecLike3, /) -> ReferenceFrame\n"
               "Create a copy of the frame with different offset value.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "offset : VecLike3\n"
               "    Offset to add to the reference frame relative to its parent.\n"
               "Returns\n"
               "-------\n"
               "ReferenceFrame\n"
               "    A copy of itself which is translated by the value of ``offset`` in\n"
               "    the parent's reference frame.\n"},
    {.ml_name = "at_time",
     .ml_meth = pyvl_reference_frame_at_time,
     .ml_flags = METH_O,
     .ml_doc = "at_time(t: float, /) -> Self\n"
               "Compute reference frame at the given time.\n"
               "\n"
               "This is used when the reference frame is moving or rotating in space.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "t : float\n"
               "    Time at which the reference frame is needed.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    New reference frame at the given time.\n"},
    {.ml_name = "angles_from_rotation",
     .ml_meth = pyvl_matrix_to_angles,
     .ml_flags = METH_O | METH_STATIC,
     .ml_doc = "angles_from_rotation(mat: array, /) -> array\n"
               "Compute rotation angles from a transformation matrix.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "mat : (3, 3) array\n"
               "    Rotation matrix to convert to the rotation angles. This is done assuming that the\n"
               "    matrix is orthogonal.\n"
               "Returns\n"
               "-------\n"
               "(3,) array"
               "    Rotation angles around the x-, y-, and z-axis which result in a transformation\n"
               "    with equal rotation matrix.\n"},
    {.ml_name = "save",
     .ml_meth = pyvl_reference_frame_save,
     .ml_flags = METH_O,
     .ml_doc = "save(hmap: HirearchicalMap, /)\n"
               "Serialize the ReferenceFrame into a HirearchicalMap.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "hmap : HirearchicalMap\n"
               "    :class:`HirearchicalMap` in which to save the reference frame into."},
    {.ml_name = "load",
     .ml_meth = (void *)pyvl_reference_frame_load,
     .ml_flags = METH_VARARGS | METH_KEYWORDS | METH_CLASS,
     .ml_doc = "load(self, hmap: HirearchicalMap, parent: ReferenceFrame | None = None/) -> Self\n"
               "Load the ReferenceFrame from a HirearchicalMap.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "hmap : HirearchicalMap\n"
               "    A :class:`HirearchicalMap`, which was created with a call to :meth:`ReferenceFrame.save`.\n"
               "parent : ReferenceFrame, optional\n"
               "    Parent of the reference frame.\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    Deserialized :class:`ReferenceFrame`.\n"},
    {.ml_name = "add_velocity",
     .ml_meth = (void *)pyvl_reference_frame_add_velocity,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "add_velocity(positions: array, velocity: array, /) -> None\n"
               "Add the velocity at the specified positions.\n"
               "\n"
               "This method exists to account for the motion of the mesh from non-stationary\n"
               "reference frames.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "positions : (N, 3) array\n"
               "   Array of :math:`N` position vectors specifying the positions where the\n"
               "   velocity should be updated.\n"
               "\n"
               "velocity : (N, 3) array\n"
               "   Array to which the velocity vectors at the specified positions should be added\n"
               "   to. These values should be added to and not just overwritten.\n"},
    {0},
};

PyDoc_STRVAR(
    pyvl_reference_frame_type_docstring,
    "ReferenceFrame(offset: VecLike3 = (0, 0, 0), theta:VecLike3 = (0, 0, 0), parent: ReferenceFrame | None = None)\n"
    "Class which is used to define position and orientation of geometry.\n"
    "\n"
    "This class represents a translation, followed by and orthonormal rotation. This transformation from a position "
    "vector"
    " :math:`\\vec{r}` in the parent reference frame to a vector :math:`\\vec{r}^\\prime` in child reference frame can"
    " be written in four steps:\n"
    "\n"
    ".. math::\n"
    "\n"
    "    \\vec{r}_1 = \\begin{bmatrix} 1 & 0 & 0 \\\\ 0 & \\cos\\theta_x & \\sin\\theta_x \\\\ 0 & -\\sin\\theta_x & "
    "\\cos\\theta_x \\end{bmatrix} \\vec{r}\n"
    "\n"
    ".. math::\n"
    "\n"
    "    \\vec{r}_2 = \\begin{bmatrix} -\\sin\\theta_y & 0 & \\cos\\theta_y \\\\ 0 & 1 & 0 \\\\ \\cos\\theta_y & 0 & "
    "\\sin\\theta_y \\end{bmatrix} \\vec{r}_1\n"
    "\n"
    ".. math::\n"
    "\n"
    "    \\vec{r}^3 = \\begin{bmatrix} \\cos\\theta_z & \\sin\\theta_z & 0 \\\\ -\\sin\\theta_z & \\cos\\theta_z "
    "& "
    "0 \\\\ 0 & 0 & 1 \\end{bmatrix} \\vec{r}_2\n"
    ".. math::\n"
    "\n"
    "    \\vec{r}^\\prime = \\vec{r}_3 + \\vec{d}\n"
    "\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "offset : VecLike3, default: (0, 0, 0)\n"
    "    Position of the reference frame's origin expressed in the parent's reference frame.\n"
    "\n"
    "theta : VecLike3, default: (0, 0, 0)\n"
    "    Rotation of the reference frame relative to its parent. The rotations are applied\n"
    "    around the x, y, and z axis in that order.\n"
    "\n");

CVL_INTERNAL
PyTypeObject pyvl_reference_frame_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pyvl.cvl.ReferenceFrame",
    .tp_basicsize = sizeof(PyVL_ReferenceFrame),
    .tp_itemsize = 0,
    .tp_repr = pyvl_reference_frame_repr,
    .tp_doc = pyvl_reference_frame_type_docstring,
    .tp_getset = pyvl_reference_frame_getset,
    .tp_methods = pyvl_reference_frame_methods,
    .tp_new = pyvl_reference_frame_new,
    .tp_dealloc = pyvl_reference_frame_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_richcompare = pyvl_reference_frame_rich_compare,
};
