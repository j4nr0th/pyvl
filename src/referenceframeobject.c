//
// Created by jan on 29.11.2024.
//
#include "referenceframeobject.h"
#include <numpy/arrayobject.h>

static PyObject *pydust_reference_frame_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    double theta_x = 0, theta_y = 0, theta_z = 0;
    double offset_x = 0, offset_y = 0, offset_z = 0;
    PyDust_ReferenceFrame *parent = nullptr;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|ddddddO!",
            (char *[8]){"theta_x", "theta_y", "theta_z", "offset_x", "offset_y", "offset_z", "parent", nullptr},
            &theta_x, &theta_y, &theta_z, &offset_x, &offset_y, &offset_z, type, &parent))
    {
        return nullptr;
    }

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
    if (this->parent)
    {
        out = PyUnicode_FromFormat("ReferenceFrame(%d, %d, %d, %d, %d, %d, parent=%R)", this->transformation.angles.x,
                                   this->transformation.angles.y, this->transformation.angles.z,
                                   this->transformation.offset.x, this->transformation.offset.y,
                                   this->transformation.offset.z, (PyObject *)this->parent);
    }
    else
    {
        out = PyUnicode_FromFormat("ReferenceFrame(%d, %d, %d, %d, %d, %d)", this->transformation.angles.x,
                                   this->transformation.angles.y, this->transformation.angles.z,
                                   this->transformation.offset.x, this->transformation.offset.y,
                                   this->transformation.offset.z);
    }
    // Py_ReprLeave(self);
    return out;
}

static inline int64_t rotate(const int64_t v, const unsigned n)
{
    return (v << n) | (v >> (8 * sizeof(v) - n));
}

static Py_hash_t pydust_reference_frame_hash(PyObject *self)
{
    const PyDust_ReferenceFrame *this = (PyDust_ReferenceFrame *)self;
    Py_hash_t h = 0;
    while (this)
    {
        h ^= rotate(*(int64_t *)&this->transformation.angles.v0, 0);
        h ^= rotate(*(int64_t *)&this->transformation.angles.v1, 4);
        h ^= rotate(*(int64_t *)&this->transformation.angles.v2, 8);
        h ^= rotate(*(int64_t *)&this->transformation.offset.v0, 12);
        h ^= rotate(*(int64_t *)&this->transformation.offset.v1, 16);
        h ^= rotate(*(int64_t *)&this->transformation.offset.v2, 24);
        this = this->parent;
    }
    return h;
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
    p_out[1] = this->transformation.offset.x;
    p_out[2] = this->transformation.offset.x;
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
    p_out[1] = this->transformation.angles.x;
    p_out[2] = this->transformation.angles.x;
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
    {.name = "rotation matrix",
     .get = pydust_reference_frame_get_rotation_matrix,
     .set = nullptr,
     .doc = "Matrix representing rotation of the reference frame.",
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
        if (PyArray_FLAGS(out_array) & NPY_ARRAY_C_CONTIGUOUS)
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

static PyObject *pydust_reference_frame_rotate_x(PyObject *self, PyObject *arg)
{
    const PyDust_ReferenceFrame *const this = (PyDust_ReferenceFrame *)self;
    const double theta = PyFloat_AsDouble(arg);
    if (!PyErr_Occurred())
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
    if (!PyErr_Occurred())
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
    if (!PyErr_Occurred())
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

static PyMethodDef pydust_reference_frame_methods[] = {
    {.ml_name = "from_parent_with_offset",
     .ml_meth = (PyCFunction)pydust_reference_frame_from_parent_with_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation to the reference frame from parent with offset."},
    {.ml_name = "from_parent_without_offset",
     .ml_meth = (PyCFunction)pydust_reference_frame_from_parent_without_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation to the reference frame from parent without offset."},
    {.ml_name = "to_parent_with_offset",
     .ml_meth = (PyCFunction)pydust_reference_frame_to_parent_with_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation from the reference frame to parent with offset."},
    {.ml_name = "to_parent_without_offset",
     .ml_meth = (PyCFunction)pydust_reference_frame_to_parent_without_offset,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Apply transformation from the reference frame to parent without offset."},
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
    {},
};

constexpr PyDoc_STRVAR(pydust_reference_frame_type_docstring,
                       "Class which is used to define position and orientation of geometry.");

CDUST_INTERNAL
PyTypeObject pydust_reference_frame_type = {
    .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "cdust.ReferenceFrame",
    .tp_basicsize = sizeof(PyDust_ReferenceFrame),
    .tp_itemsize = 0,
    .tp_repr = pydust_reference_frame_repr,
    .tp_hash = pydust_reference_frame_hash,
    .tp_doc = pydust_reference_frame_type_docstring,
    .tp_getset = pydust_reference_frame_getset,
    .tp_methods = pydust_reference_frame_methods,
    .tp_new = pydust_reference_frame_new,
    .tp_dealloc = pydust_reference_frame_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_IMMUTABLETYPE,
};
