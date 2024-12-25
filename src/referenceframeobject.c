//
// Created by jan on 29.11.2024.
//
#include "referenceframeobject.h"
#include <numpy/arrayobject.h>

static PyObject *pydust_reference_frame_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    double theta_x = 0, theta_y = 0, theta_z = 0;
    double offset_x = 0, offset_y = 0, offset_z = 0;
    PyDust_ReferenceFrame *parent;
    PyObject *p = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|(ddd)(ddd)O", (char *[4]){"offset", "theta", "parent", nullptr},
                                     &offset_x, &offset_y, &offset_z, &theta_x, &theta_y, &theta_z, &p))
    {
        return nullptr;
    }
    if (p == nullptr || Py_IsNone(p))
    {
        parent = nullptr;
    }
    else if (!PyObject_TypeCheck(p, &pydust_reference_frame_type))
    {
        PyErr_Format(PyExc_TypeError, "Argument \"parent\" must be a ReferenceFrame object, but it was %R", Py_TYPE(p));
        return nullptr;
    }
    else
    {
        parent = (PyDust_ReferenceFrame *)p;
    }
    theta_x = clamp_angle_to_range(theta_x);
    theta_y = clamp_angle_to_range(theta_y);
    theta_z = clamp_angle_to_range(theta_z);

    PyDust_ReferenceFrame *const this = (PyDust_ReferenceFrame *)type->tp_alloc(type, 0);
    if (!this)
    {
        return nullptr;
    }
    Py_XINCREF(parent);

    this->parent = parent;
    this->transformation = (transformation_t){
        .angles = {.x = theta_x, .y = theta_y, .z = theta_z},
        .offset = {.x = offset_x, .y = offset_y, .z = offset_z},
    };
    return (PyObject *)this;
}

static void pydust_reference_frame_dealloc(PyObject *self)
{
    PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    Py_XDECREF(this->parent);
    Py_TYPE(this)->tp_free(this);
}

static PyObject *pydust_reference_frame_repr(PyObject *self)
{
    // Recursion should never happen, since there's no real way to make a cycle due to being unable to set
    // the parent in Python code.

    // const int repr_res = Py_ReprEnter(self);
    // if (repr_res < 0) return nullptr;
    // if (repr_res > 0) return PyUnicode_FromString("...");

    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    PyObject *out;
    unsigned len = snprintf(nullptr, 0, "(%g, %g, %g), (%g, %g, %g)", this->transformation.angles.x,
                            this->transformation.angles.y, this->transformation.angles.z, this->transformation.offset.x,
                            this->transformation.offset.y, this->transformation.offset.z);
    char *buffer = PyMem_Malloc((len + 1) * sizeof *buffer);
    if (!buffer)
        return nullptr;
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

static PyObject *pydust_reference_frame_get_parent(PyObject *self, void *Py_UNUSED(closure))
{
    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    if (!this->parent)
    {
        Py_RETURN_NONE;
    }
    Py_INCREF(this->parent);
    return (PyObject *)this->parent;
}

static PyObject *pydust_reference_frame_get_offset(PyObject *self, void *Py_UNUSED(closure))
{
    constexpr npy_intp dim = 3;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
    if (!out)
        return nullptr;
    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    double *const p_out = PyArray_DATA(out);
    if (!p_out)
    {
        // I don't think this would ever even happen.
        Py_DECREF(out);
        return nullptr;
    }
    p_out[0] = this->transformation.offset.x;
    p_out[1] = this->transformation.offset.y;
    p_out[2] = this->transformation.offset.z;
    return (PyObject *)out;
}

static PyObject *pydust_reference_frame_get_angles(PyObject *self, void *Py_UNUSED(closure))
{
    constexpr npy_intp dim = 3;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
    if (!out)
        return nullptr;
    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    double *const p_out = PyArray_DATA(out);
    if (!p_out)
    {
        // I don't think this would ever even happen.
        Py_DECREF(out);
        return nullptr;
    }
    p_out[0] = this->transformation.angles.x;
    p_out[1] = this->transformation.angles.y;
    p_out[2] = this->transformation.angles.z;
    return (PyObject *)out;
}

static PyObject *pydust_reference_frame_get_rotation_matrix(PyObject *self, void *Py_UNUSED(closure))
{
    constexpr npy_intp dims[2] = {3, 3};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
        return nullptr;
    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    double *const p_out = PyArray_DATA(out);
    if (!p_out)
    {
        // I don't think this would ever even happen.
        Py_DECREF(out);
        return nullptr;
    }
    const real3x3_t mat = real3x3_from_angles(this->transformation.angles);
    for (unsigned i = 0; i < 9; ++i)
    {
        p_out[i] = mat.data[i];
    }
    return (PyObject *)out;
}

static PyObject *pydust_reference_frame_get_rotation_matrix_inverse(PyObject *self, void *Py_UNUSED(closure))
{
    constexpr npy_intp dims[2] = {3, 3};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
        return nullptr;
    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    double *const p_out = PyArray_DATA(out);
    if (!p_out)
    {
        // I don't think this would ever even happen.
        Py_DECREF(out);
        return nullptr;
    }
    const real3x3_t mat = real3x3_inverse_from_angles(this->transformation.angles);
    for (unsigned i = 0; i < 9; ++i)
    {
        p_out[i] = mat.data[i];
    }
    return (PyObject *)out;
}

static PyObject *pydust_reference_frame_get_parents(PyObject *self, void *Py_UNUSED(closure))
{
    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    unsigned parent_cnt = 0;
    for (PyDust_ReferenceFrame *p = this->parent; p; p = p->parent)
        parent_cnt += 1;
    PyObject *out = PyTuple_New(parent_cnt);
    unsigned i = 0;
    for (PyDust_ReferenceFrame *p = this->parent; p; p = p->parent)
    {
        Py_INCREF(p);
        PyTuple_SET_ITEM(out, i, p);
        i += 1;
    }
    return out;
}

static PyObject *pydust_reference_frame_rich_compare(PyObject *self, PyObject *other, const int op)
{
    constexpr real_t tol = 1e-10;
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    if (!PyType_CheckExact(other, &pydust_reference_frame_type))
    {
        Py_RETURN_FALSE;
    }
    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    const PyDust_ReferenceFrame *that = (PyDust_ReferenceFrame *)other;

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

static PyGetSetDef pydust_reference_frame_getset[] = {
    {.name = "parent",
     .get = pydust_reference_frame_get_parent,
     .set = nullptr,
     .doc = "What frame it is relative to.",
     .closure = nullptr},
    {.name = "offset",
     .get = pydust_reference_frame_get_offset,
     .set = nullptr,
     .doc = "Vector determining the offset of the reference frame in parent's frame.",
     .closure = nullptr},
    {.name = "angles",
     .get = pydust_reference_frame_get_angles,
     .set = nullptr,
     .doc = "Vector determining the offset of the reference frame in parent's frame.",
     .closure = nullptr},
    {.name = "rotation_matrix",
     .get = pydust_reference_frame_get_rotation_matrix,
     .set = nullptr,
     .doc = "Matrix representing rotation of the reference frame.",
     .closure = nullptr},
    {.name = "rotation_matrix_inverse",
     .get = pydust_reference_frame_get_rotation_matrix_inverse,
     .set = nullptr,
     .doc = "Matrix representing inverse rotation of the reference frame.",
     .closure = nullptr},
    {.name = "parents",
     .get = pydust_reference_frame_get_parents,
     .set = nullptr,
     .doc = "Tuple of all parents of this reference frame.",
     .closure = nullptr},
    {},
};

static bool prepare_for_transformation(Py_ssize_t nargs, PyObject *const args[static nargs], PyArrayObject **p_in,
                                       PyArrayObject **p_out, npy_intp *p_dim, const npy_intp **p_dims)
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
                                                                     NPY_ARRAY_C_CONTIGUOUS, nullptr);
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
        out_array = (PyArrayObject *)PyArray_NewLikeArray(in_array, NPY_CORDER, nullptr, 0);
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

static PyObject *pydust_reference_frame_from_parent_with_offset(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return nullptr;
    }

    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    const real3x3_t mat = real3x3_from_angles(this->transformation.angles);
    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < dim_in - 1; ++i)
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

static PyObject *pydust_reference_frame_from_parent_without_offset(PyObject *self, PyObject *const *args,
                                                                   Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return nullptr;
    }

    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    const real3x3_t mat = real3x3_from_angles(this->transformation.angles);
    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < dim_in - 1; ++i)
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

static PyObject *pydust_reference_frame_to_parent_with_offset(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return nullptr;
    }

    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    const real3x3_t mat = real3x3_inverse_from_angles(this->transformation.angles);
    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < dim_in - 1; ++i)
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

static PyObject *pydust_reference_frame_to_parent_without_offset(PyObject *self, PyObject *const *args,
                                                                 Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return nullptr;
    }

    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    const real3x3_t mat = real3x3_inverse_from_angles(this->transformation.angles);
    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < dim_in - 1; ++i)
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

static PyObject *pydust_reference_frame_from_global_with_offset(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return nullptr;
    }

    const PyDust_ReferenceFrame *const this = (PyDust_ReferenceFrame *)self;
    real3x3_t mat = real3x3_from_angles(this->transformation.angles);
    real3_t off = this->transformation.offset;
    for (const PyDust_ReferenceFrame *p = this; p->parent; p = p->parent)
    {
        merge_transformations(real3x3_from_angles(p->transformation.angles), p->transformation.offset, mat, off, &mat,
                              &off);
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < dim_in - 1; ++i)
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

static PyObject *pydust_reference_frame_from_global_without_offset(PyObject *self, PyObject *const *args,
                                                                   Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return nullptr;
    }

    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    real3x3_t mat = real3x3_from_angles(this->transformation.angles);
    for (const PyDust_ReferenceFrame *p = this; p->parent; p = p->parent)
    {
        mat = real3x3_matmul(real3x3_from_angles(p->transformation.angles), mat);
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < dim_in - 1; ++i)
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

static PyObject *pydust_reference_frame_to_global_with_offset(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return nullptr;
    }

    const PyDust_ReferenceFrame *const this = (PyDust_ReferenceFrame *)self;
    real3x3_t mat = real3x3_inverse_from_angles(this->transformation.angles);
    real3_t off = this->transformation.offset;
    for (const PyDust_ReferenceFrame *p = this; p->parent; p = p->parent)
    {
        merge_transformations_reverse(real3x3_inverse_from_angles(p->transformation.angles), p->transformation.offset,
                                      mat, off, &mat, &off);
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < dim_in - 1; ++i)
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

static PyObject *pydust_reference_frame_to_global_without_offset(PyObject *self, PyObject *const *args,
                                                                 Py_ssize_t nargs)
{
    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    if (!prepare_for_transformation(nargs, args, &in_array, &out_array, &dim_in, &dims_in))
    {
        return nullptr;
    }

    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    real3x3_t mat = real3x3_inverse_from_angles(this->transformation.angles);
    for (const PyDust_ReferenceFrame *p = this; p->parent; p = p->parent)
    {
        mat = real3x3_matmul(mat, real3x3_inverse_from_angles(p->transformation.angles));
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < dim_in - 1; ++i)
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

static PyObject *pydust_reference_frame_rotate_x(PyObject *self, PyObject *arg)
{
    const PyDust_ReferenceFrame *const this = (PyDust_ReferenceFrame *)self;
    const double theta = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
        return nullptr;

    PyDust_ReferenceFrame *const new =
        (PyDust_ReferenceFrame *)pydust_reference_frame_type.tp_alloc(&pydust_reference_frame_type, 0);
    if (!new)
        return nullptr;
    new->transformation = this->transformation;
    new->parent = this->parent;
    Py_XINCREF(this->parent);
    new->transformation.angles.x = clamp_angle_to_range(new->transformation.angles.x + theta);
    return (PyObject *)new;
}

static PyObject *pydust_reference_frame_rotate_y(PyObject *self, PyObject *arg)
{
    const PyDust_ReferenceFrame *const this = (PyDust_ReferenceFrame *)self;
    const double theta = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
        return nullptr;

    PyDust_ReferenceFrame *const new =
        (PyDust_ReferenceFrame *)pydust_reference_frame_type.tp_alloc(&pydust_reference_frame_type, 0);
    if (!new)
        return nullptr;
    new->transformation = this->transformation;
    new->parent = this->parent;
    Py_XINCREF(this->parent);
    new->transformation.angles.y = clamp_angle_to_range(new->transformation.angles.y + theta);
    return (PyObject *)new;
}

static PyObject *pydust_reference_frame_rotate_z(PyObject *self, PyObject *arg)
{
    const PyDust_ReferenceFrame *const this = (PyDust_ReferenceFrame *)self;
    const double theta = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
        return nullptr;

    PyDust_ReferenceFrame *const new =
        (PyDust_ReferenceFrame *)pydust_reference_frame_type.tp_alloc(&pydust_reference_frame_type, 0);
    if (!new)
        return nullptr;
    new->transformation = this->transformation;
    new->parent = this->parent;
    Py_XINCREF(this->parent);
    new->transformation.angles.z = clamp_angle_to_range(new->transformation.angles.z + theta);
    return (PyObject *)new;
}

static PyObject *pydust_reference_frame_with_offset(PyObject *self, PyObject *arg)
{
    const PyDust_ReferenceFrame *const this = (PyDust_ReferenceFrame *)self;
    PyArrayObject *const off_array = (PyArrayObject *)PyArray_FromAny(arg, PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
                                                                      NPY_ARRAY_C_CONTIGUOUS, nullptr);
    if (!off_array)
        return nullptr;
    const npy_intp *p_n = PyArray_DIMS(off_array);
    if (*p_n != 3)
    {
        PyErr_Format(PyExc_ValueError, "Input array must have 3 element, instead %u were given.", (unsigned)*p_n);
        Py_DECREF(off_array);
        return nullptr;
    }
    const npy_float64 *p_in = PyArray_DATA(off_array);
    const real3_t new_offset = {.v0 = p_in[0], .v1 = p_in[1], .v2 = p_in[2]};
    Py_DECREF(off_array);

    PyDust_ReferenceFrame *const new =
        (PyDust_ReferenceFrame *)pydust_reference_frame_type.tp_alloc(&pydust_reference_frame_type, 0);
    if (!new)
        return nullptr;
    new->transformation = this->transformation;
    new->parent = this->parent;
    Py_XINCREF(this->parent);
    new->transformation.offset = new_offset;
    return (PyObject *)new;
}

static PyObject *pydust_reference_frame_at_time(PyObject *self, PyObject *arg)
{
    const double time = PyFloat_AsDouble(arg);
    if (PyErr_Occurred())
        return nullptr;
    (void)time;
    PyDust_ReferenceFrame *const new =
        (PyDust_ReferenceFrame *)pydust_reference_frame_type.tp_alloc(&pydust_reference_frame_type, 0);
    if (!new)
        return nullptr;
    const PyDust_ReferenceFrame *const this = (PyDust_ReferenceFrame *)self;
    new->transformation = this->transformation;
    if (this->parent)
    {
        //  Evolve the parent.
        new->parent = (PyDust_ReferenceFrame *)pydust_reference_frame_at_time((PyObject *)this->parent, arg);
    }
    else
    {
        new->parent = nullptr;
    }
    return (PyObject *)new;
}

static PyObject *pydust_matrix_to_angles(PyObject *Py_UNUSED(module), PyObject *arg)
{
    PyArrayObject *const array = (PyArrayObject *)PyArray_FROMANY(arg, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!array)
        return nullptr;
    const npy_intp *const dims = PyArray_DIMS(array);
    if (dims[0] != 3 || dims[1] != 3)
    {
        PyErr_Format(PyExc_ValueError, "Array was not a (3, 3) array, instead it was (%u, %u).", (unsigned)dims[0],
                     (unsigned)dims[1]);
        return nullptr;
    }
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, (const npy_intp[1]){3}, NPY_DOUBLE);
    if (!out)
    {
        Py_DECREF(array);
        return nullptr;
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

static PyObject *pydust_reference_frame_save(PyObject *self, PyObject *arg)
{
    if (!PyMapping_Check(arg))
    {
        PyErr_Format(PyExc_TypeError, "The input parameter is not a mapping.");
        return nullptr;
    }
    PyObject *const off_array = pydust_reference_frame_get_offset(self, nullptr);
    PyObject *const rot_array = pydust_reference_frame_get_angles(self, nullptr);
    if (!off_array || !rot_array)
    {
        Py_XDECREF(off_array);
        Py_XDECREF(rot_array);
        return nullptr;
    }
    // if (this->parent)
    // {
    //     PyObject *const out_group = PyObject_CallMethod(arg, "create_group", "s", "parent");
    //     if (!out_group)
    //     {
    //         Py_DECREF(off_array);
    //         Py_DECREF(rot_array);
    //         return nullptr;
    //     }
    //     PyObject *const res = pydust_reference_frame_save((PyObject *)this->parent, out_group);
    //     Py_DECREF(out_group);
    //     Py_XDECREF(res);
    //     if (!res)
    //         return nullptr;
    // }
    const int res1 = PyMapping_SetItemString(arg, "offset", off_array);
    Py_DECREF(off_array);
    const int res2 = PyMapping_SetItemString(arg, "angles", rot_array);
    Py_DECREF(rot_array);
    if (res1 < 0 || res2 < 0)
        return nullptr;
    PyObject *type_name = PyUnicode_FromString(pydust_reference_frame_type.tp_name);
    if (!type_name)
        return nullptr;
    const int res3 = PyMapping_SetItemString(arg, "type", type_name);
    Py_DECREF(type_name);
    if (res3 < 0)
        return nullptr;
    Py_RETURN_NONE;
}

static PyObject *pydust_reference_frame_load(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *group, *parent = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char *[3]){"group", "parent", nullptr}, &group, &parent))
    {
        return nullptr;
    }
    if (parent)
    {
        if (Py_IsNone(parent))
        {
            parent = nullptr;
        }
        else if (!PyObject_TypeCheck(parent, &pydust_reference_frame_type))
        {
            PyErr_Format(PyExc_TypeError, "Parent was neither None nor a ReferenceFrame (instead it was %R).",
                         Py_TYPE(parent));
            return nullptr;
        }
    }
    if (!PyMapping_Check(group))
    {
        PyErr_Format(PyExc_TypeError, "The input parameter is not a mapping.");
        return nullptr;
    }
    // PyObject *const type_name = PyMapping_GetItemString(group, "type");
    // if (!type_name)
    //     return nullptr;
    // Py_ssize_t len;
    // const char *const full_name = PyUnicode_AsUTF8AndSize(type_name, &len);
    // if (!full_name)
    // {
    //     Py_DECREF(type_name);
    //     return nullptr;
    // }
    // const char *const split = strrchr(full_name, '.');
    // if (!split)
    // {
    //     Py_DECREF(type_name);
    //     PyErr_Format(PyExc_ValueError, "The type name did not contain any \".\" characters.");
    //     return nullptr;
    // }
    // const unsigned mod_len = split - full_name;
    // char *const buffer = PyMem_Malloc(sizeof(*buffer) * (mod_len));
    // if (!buffer)
    // {
    //     Py_DECREF(type_name);
    //     return nullptr;
    // }
    // memcpy(buffer, full_name, mod_len - 1);
    // buffer[mod_len - 1] = 0;
    // PyObject *module = PyImport_ImportModule(buffer);
    // PyMem_Free(buffer);
    // if (!module)
    // {
    //     Py_DECREF(type_name);
    //     return nullptr;
    // }
    PyTypeObject *type = (PyTypeObject *)self; // PyObject_GetAttrString(module, split + 1);
    // Py_DECREF(type_name);
    // if (!type)
    //     return nullptr;

    PyDust_ReferenceFrame *const this = (PyDust_ReferenceFrame *)type->tp_alloc(type, 0);
    // Py_DECREF(type);
    if (!this)
        return nullptr;
    PyObject *const off_val = PyMapping_GetItemString(group, "offset");
    PyObject *const rot_val = PyMapping_GetItemString(group, "angles");
    if (!off_val || !rot_val)
    {
        Py_XDECREF(off_val);
        Py_XDECREF(rot_val);
        return nullptr;
    }
    PyArrayObject *const off_array = (PyArrayObject *)PyArray_FromAny(
        off_val, PyArray_DescrFromType(NPY_FLOAT64), 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, nullptr);
    Py_DECREF(off_val);
    PyArrayObject *const rot_array = (PyArrayObject *)PyArray_FromAny(
        rot_val, PyArray_DescrFromType(NPY_FLOAT64), 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, nullptr);
    Py_DECREF(rot_val);
    if (!off_array || !rot_array)
    {
        Py_XDECREF(off_array);
        Py_XDECREF(rot_array);
        Py_DECREF(this);
        return nullptr;
    }
    if (PyArray_SIZE(off_array) != 3)
    {
        PyErr_Format(PyExc_ValueError, "Offset array did not have 3 elements, but had %u instead.",
                     (unsigned)PyArray_SIZE(off_array));
        Py_DECREF(off_array);
        Py_DECREF(rot_array);
        Py_DECREF(this);
        return nullptr;
    }
    if (PyArray_SIZE(rot_array) != 3)
    {
        PyErr_Format(PyExc_ValueError, "Angle array did not have 3 elements, but had %u instead.",
                     (unsigned)PyArray_SIZE(rot_array));
        Py_DECREF(off_array);
        Py_DECREF(rot_array);
        Py_DECREF(this);
        return nullptr;
    }

    const npy_float64 *const offset_ptr = PyArray_DATA(off_array);
    const npy_float64 *const angles_ptr = PyArray_DATA(rot_array);

    this->transformation.offset = (real3_t){.x = offset_ptr[0], .y = offset_ptr[1], .z = offset_ptr[2]};
    this->transformation.angles = (real3_t){.x = angles_ptr[0], .y = angles_ptr[1], .z = angles_ptr[2]};

    Py_DECREF(off_array);
    Py_DECREF(rot_array);
    this->parent = (PyDust_ReferenceFrame *)parent;
    Py_XINCREF(parent);

    return (PyObject *)this;
}

static PyMethodDef pydust_reference_frame_methods[] = {
    {.ml_name = "from_parent_with_offset",
     .ml_meth = (void *)pydust_reference_frame_from_parent_with_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation to the reference frame from parent with offset."},
    {.ml_name = "from_parent_without_offset",
     .ml_meth = (void *)pydust_reference_frame_from_parent_without_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation to the reference frame from parent without offset."},
    {.ml_name = "to_parent_with_offset",
     .ml_meth = (void *)pydust_reference_frame_to_parent_with_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation from the reference frame to parent with offset."},
    {.ml_name = "to_parent_without_offset",
     .ml_meth = (void *)pydust_reference_frame_to_parent_without_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation from the reference frame to parent without offset."},
    {.ml_name = "from_global_with_offset",
     .ml_meth = (void *)pydust_reference_frame_from_global_with_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation to the reference frame from global with offset."},
    {.ml_name = "from_global_without_offset",
     .ml_meth = (void *)pydust_reference_frame_from_global_without_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation to the reference frame from global without offset."},
    {.ml_name = "to_global_with_offset",
     .ml_meth = (void *)pydust_reference_frame_to_global_with_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation from the reference frame to global with offset."},
    {.ml_name = "to_global_without_offset",
     .ml_meth = (void *)pydust_reference_frame_to_global_without_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation from the reference frame to global without offset."},
    {.ml_name = "rotate_x",
     .ml_meth = pydust_reference_frame_rotate_x,
     .ml_flags = METH_O,
     .ml_doc = "Create a copy of the frame rotated around the x-axis."},
    {.ml_name = "rotate_y",
     .ml_meth = pydust_reference_frame_rotate_y,
     .ml_flags = METH_O,
     .ml_doc = "Create a copy of the frame rotated around the y-axis."},
    {.ml_name = "rotate_z",
     .ml_meth = pydust_reference_frame_rotate_z,
     .ml_flags = METH_O,
     .ml_doc = "Create a copy of the frame rotated around the z-axis."},
    {.ml_name = "with_offset",
     .ml_meth = pydust_reference_frame_with_offset,
     .ml_flags = METH_O,
     .ml_doc = "Create a copy of the frame with different offset value."},
    {.ml_name = "at_time",
     .ml_meth = pydust_reference_frame_at_time,
     .ml_flags = METH_O,
     .ml_doc = "Compute reference frame at the given time.\n"
               "\n"
               "This is useful when the reference frame is moving or rotating in space.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "t : float\n"
               "    Time at which the reference frame is needed.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "ReferenceFrame\n"
               "    New reference frame."},
    {.ml_name = "angles_from_rotation",
     .ml_meth = pydust_matrix_to_angles,
     .ml_flags = METH_O | METH_STATIC,
     .ml_doc = "Compute rotation angles from a transformation matrix."},
    {.ml_name = "save",
     .ml_meth = pydust_reference_frame_save,
     .ml_flags = METH_O,
     .ml_doc = "Serialize the ReferenceFrame into a HDF5 group."},
    {.ml_name = "load",
     .ml_meth = (void *)pydust_reference_frame_load,
     .ml_flags = METH_VARARGS | METH_KEYWORDS | METH_CLASS,
     .ml_doc = "Load the ReferenceFrame from a HDF5 group."},
    {},
};

constexpr PyDoc_STRVAR(pydust_reference_frame_type_docstring,
                       "Class which is used to define position and orientation of geometry.");

CDUST_INTERNAL
PyTypeObject pydust_reference_frame_type = {
    .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pydust.cdust.ReferenceFrame",
    .tp_basicsize = sizeof(PyDust_ReferenceFrame),
    .tp_itemsize = 0,
    .tp_repr = pydust_reference_frame_repr,
    .tp_doc = pydust_reference_frame_type_docstring,
    .tp_getset = pydust_reference_frame_getset,
    .tp_methods = pydust_reference_frame_methods,
    .tp_new = pydust_reference_frame_new,
    .tp_dealloc = pydust_reference_frame_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_richcompare = pydust_reference_frame_rich_compare,
};
