#include "time_dependent_vector.h"

#include <numpy/arrayobject.h>

// Must be after the array include
#include <cpyutl.h>

bool pyvl_vec_time_dependent_init(pyvl_vec_time_dependent_t *const field, PyObject *value, const char *field_name)
{
    if (value == NULL || Py_IsNone(value))
    {
        field->type = PYVL_VEC_CONSTANT;
        field->value.constant = (real3_t){0};
        return true;
    }

    if (PyCallable_Check(value))
    {
        field->type = PYVL_VEC_CALLABLE;
        field->value.callable = value;
        Py_INCREF(value);
        return true;
    }

    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FromAny(value, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!arr)
    {
        PyErr_Format(PyExc_TypeError, "%s must be a callable or a sequence of 3 numbers.", field_name);
        return false;
    }
    const npy_intp n = PyArray_SIZE(arr);
    if (n != 3)
    {
        Py_DECREF(arr);
        PyErr_Format(PyExc_ValueError, "Expected 3 elements, got %d.", (int)n);
        return false;
    }
    const double *const p = PyArray_DATA(arr);
    field->type = PYVL_VEC_CONSTANT;
    field->value.constant = (real3_t){.x = p[0], .y = p[1], .z = p[2]};
    Py_DECREF(arr);
    return true;
}

static PyArrayObject *pyvl_vec_time_dependent_eval_callable(const pyvl_vec_time_dependent_t *field,
                                                            PyObject *const time_arg)
{
    PyObject *const res = PyObject_Vectorcall(field->value.callable, (PyObject *const[1]){time_arg}, 1, NULL);
    if (!res)
        return NULL;

    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(res, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    Py_DECREF(res);

    if (!arr)
    {
        PyErr_SetString(PyExc_TypeError, "Callable must return a sequence of 3 numbers.");
        return NULL;
    }
    const npy_intp n = PyArray_SIZE(arr);
    if (n != 3)
    {
        Py_DECREF(arr);
        PyErr_Format(PyExc_ValueError, "Expected 3 elements, got %d.", (int)n);
        return NULL;
    }
    return arr;
}

bool pyvl_vec_time_dependent_eval_c(const pyvl_vec_time_dependent_t *const field, PyObject *const time_arg,
                                    real3_t *const value)
{
    if (field->type == PYVL_VEC_CONSTANT)
    {
        *value = field->value.constant;
        return true;
    }

    PyArrayObject *const arr = pyvl_vec_time_dependent_eval_callable(field, time_arg);
    if (!arr)
        return false;

    *value = *(real3_t *)PyArray_DATA(arr);
    Py_DECREF(arr);
    return true;
}

PyObject *pyvl_vec_time_dependent_eval_py(const pyvl_vec_time_dependent_t *const field, PyObject *const time_arg)
{
    if (field->type == PYVL_VEC_CALLABLE)
        return (PyObject *)pyvl_vec_time_dependent_eval_callable(field, time_arg);

    static const npy_intp size = 3;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if (!out)
        return NULL;
    *(real3_t *)PyArray_DATA(out) = field->value.constant;
    return (PyObject *)out;
}

void pyvl_vec_time_dependent_clear(pyvl_vec_time_dependent_t *const field)
{
    if (field->type == PYVL_VEC_CALLABLE)
        Py_DECREF(field->value.callable);
    field->type = PYVL_VEC_CONSTANT;
    field->value.constant = (real3_t){0};
}

bool pyvl_vec_time_dependent_serialize(const pyvl_vec_time_dependent_t *const entry, const char *const key,
                                       PyObject *const hmap, PyObject *const serializer)
{
    PyObject *serial = NULL;
    const npy_intp sz = 3;
    switch (entry->type)
    {
    case PYVL_VEC_CONSTANT:
        serial = PyArray_SimpleNew(1, &sz, NPY_DOUBLE);
        if (serial)
            *(real3_t *)PyArray_DATA((PyArrayObject *)serial) = entry->value.constant;
        break;
    case PYVL_VEC_CALLABLE:
        serial = PyObject_Vectorcall(serializer, &entry->value.callable, 1, NULL);
        if (serial && !PyUnicode_Check(serial))
        {
            PyErr_Format(PyExc_TypeError, "Serializer did not return a string, but a %s", Py_TYPE(serial)->tp_name);
            Py_DECREF(serial);
            serial = NULL;
        }
        break;
    }

    if (!serial)
        return false;

    const int insertion_res = PyMapping_SetItemString(hmap, key, serial);
    Py_DECREF(serial);
    return insertion_res == 0;
}

bool pyvl_vec_time_dependent_deserialize(const char *const key, PyObject *const hmap, PyObject *const deserializer,
                                         pyvl_vec_time_dependent_t *const entry)
{
    PyObject *const value = PyMapping_GetItemString(hmap, key);
    if (!value)
        return false;

    if (PyArray_Check(value))
    {
        const PyArrayObject *const arr = (PyArrayObject *)value;
        if (check_input_array(arr, 1, (npy_intp[1]){3}, NPY_DOUBLE, 0, key) < 0)
        {
            Py_DECREF(value);
            return false;
        }

        *entry = (pyvl_vec_time_dependent_t){.type = PYVL_VEC_CONSTANT,
                                             .value = {.constant = {.x = *(double *)PyArray_GETPTR1(arr, 0),
                                                                    .y = *(double *)PyArray_GETPTR1(arr, 1),
                                                                    .z = *(double *)PyArray_GETPTR1(arr, 2)}}};
        Py_DECREF(value);
        return true;
    }

    if (PyUnicode_Check(value))
    {
        PyObject *const deserialized = PyObject_Vectorcall(deserializer, &value, 1, NULL);
        Py_DECREF(value);
        if (!deserialized)
            return false;

        if (!PyCallable_Check(deserialized))
        {
            PyErr_Format(PyExc_TypeError, "The deserializer did not produce a callable, but a %s",
                         Py_TYPE(deserialized)->tp_name);
            Py_DECREF(deserialized);
            return false;
        }

        *entry = (pyvl_vec_time_dependent_t){.type = PYVL_VEC_CALLABLE, .value = {.callable = deserialized}};
        return true;
    }

    PyErr_Format(PyExc_TypeError, "Entry %s was neither a string nor an array and was instead %s.", key,
                 Py_TYPE(value)->tp_name);
    Py_DECREF(value);
    return false;
}
