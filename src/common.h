//
// Created by jan on 14.12.2024.
//

#ifndef PYVL_COMMON_H
#define PYVL_COMMON_H

static inline PyArrayObject *pyvl_ensure_array(PyObject *arr, unsigned n_dims,
                                               const npy_intp CVL_ARRAY_ARG(dims, static n_dims), int flags, int type,
                                               const char *array_name)
{
    if (!PyArray_Check(arr))
    {
        PyErr_Format(PyExc_TypeError, "%s was not an array, but was instead %R", array_name, PyObject_Type(arr));
        return NULL;
    }

    PyArrayObject *const this = (PyArrayObject *)arr;
    if (n_dims != 0)
    {
        if (PyArray_NDIM(this) != (int)n_dims)
        {
            PyErr_Format(PyExc_ValueError, "%s did not have the expected number of axis (%u required, %u found).",
                         array_name, n_dims, (unsigned)PyArray_NDIM(this));
            return NULL;
        }
        const npy_intp *real_dims = PyArray_DIMS(this);
        for (unsigned i = 0; i < n_dims; ++i)
        {
            if (dims[i] != 0 && real_dims[i] != dims[i])
            {
                PyErr_Format(PyExc_ValueError,
                             "%s did not have the expected shape (axis %u should have the size"
                             " of %u, but was instead %u).",
                             array_name, i, dims[i], (unsigned)real_dims[i]);
                return NULL;
            }
        }
    }
    if (flags)
    {
        const int real_flags = PyArray_FLAGS(this);
        if ((flags & NPY_ARRAY_C_CONTIGUOUS) && !(real_flags & NPY_ARRAY_C_CONTIGUOUS))
        {
            PyErr_Format(PyExc_ValueError, "%s was not C-contiguous.", array_name);
            return NULL;
        }
        if ((flags & NPY_ARRAY_WRITEABLE) && !(real_flags & NPY_ARRAY_WRITEABLE))
        {
            PyErr_Format(PyExc_ValueError, "%s was not writable.", array_name);
            return NULL;
        }
        if ((flags & NPY_ARRAY_ALIGNED) && !(real_flags & NPY_ARRAY_ALIGNED))
        {
            PyErr_Format(PyExc_ValueError, "%s was not aligned.", array_name);
            return NULL;
        }
    }
    if (type > 0 && PyArray_TYPE(this) != type)
    {
        PyErr_Format(PyExc_ValueError, "%s did not have the correct data type.", array_name);
        return NULL;
    }

    return this;
}

#endif // PYVL_COMMON_H
