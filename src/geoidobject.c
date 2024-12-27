//
// Created by jan on 23.11.2024.
//

#include "geoidobject.h"

PyObject *geoid_repr(PyObject *self)
{
    const PyVL_GeoIDObject *this = (PyVL_GeoIDObject *)self;
    return PyUnicode_FromFormat("GeoID(%u, %u)", (unsigned)this->id.value, (unsigned)this->id.orientation);
}

PyObject *geoid_str(PyObject *self)
{
    const PyVL_GeoIDObject *this = (PyVL_GeoIDObject *)self;
    return PyUnicode_FromFormat("%c%u", (unsigned)this->id.orientation ? '-' : '+', (unsigned)this->id.value);
}

static PyObject *geoid_get_orientation(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_GeoIDObject *this = (PyVL_GeoIDObject *)self;
    return PyBool_FromLong(this->id.orientation);
}

static int geoid_set_orientation(PyObject *self, PyObject *value, void *Py_UNUSED(closure))
{
    PyVL_GeoIDObject *this = (PyVL_GeoIDObject *)self;
    const int val = PyObject_IsTrue(value);
    if (val < 0)
    {
        return val;
    }
    this->id.orientation = val == 0 ? 0 : 1;
    return 0;
}

static PyObject *geoid_get_index(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_GeoIDObject *this = (PyVL_GeoIDObject *)self;
    return PyLong_FromUnsignedLong(this->id.value);
}

static int geoid_set_index(PyObject *self, PyObject *value, void *Py_UNUSED(closure))
{
    PyVL_GeoIDObject *this = (PyVL_GeoIDObject *)self;
    const unsigned v = PyLong_AsUnsignedLong(value);
    if (PyErr_Occurred())
    {
        return -1;
    }
    this->id.value = v;
    return 0;
}

static PyGetSetDef geoid_getset[] = {
    {.name = "orientation",
     .get = geoid_get_orientation,
     .set = geoid_set_orientation,
     .doc = "Orientation of the object referenced by id.",
     .closure = NULL},
    {.name = "index",
     .get = geoid_get_index,
     .set = geoid_set_index,
     .doc = "Index of the object referenced by id.",
     .closure = NULL},
    {}, // sentinel
};

static PyObject *geoid_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    unsigned long value;
    int orientation = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "I|p", (char *[3]){"index", "orientation", NULL}, &value,
                                     &orientation))
    {
        return NULL;
    }

    PyVL_GeoIDObject *const this = (PyVL_GeoIDObject *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;

    this->id.orientation = orientation;
    this->id.value = value;

    return (PyObject *)this;
}

static PyObject *geoid_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyVL_GeoIDObject *const this = (PyVL_GeoIDObject *)self;
    if (!PyObject_TypeCheck(other, &pyvl_geoid_type))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyVL_GeoIDObject *const that = (PyVL_GeoIDObject *)other;
    const bool val = this->id.orientation == that->id.orientation && this->id.value == that->id.value;
    if (op == Py_NE)
    {
        return PyBool_FromLong(!val);
    }
    return PyBool_FromLong(val);
}

PyDoc_STRVAR(geoid_type_docstring, "Class used to refer to topological objects with orientation.\n");

PyTypeObject pyvl_geoid_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pyvl.cvl.GeoID",
    .tp_basicsize = sizeof(PyVL_GeoIDObject),
    .tp_itemsize = 0,
    .tp_getset = geoid_getset,
    .tp_repr = geoid_repr,
    .tp_str = geoid_str,
    .tp_doc = geoid_type_docstring,
    .tp_new = geoid_new,
    .tp_richcompare = geoid_rich_compare,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
};
