//
// Created by jan on 23.11.2024.
//

#include "lineobject.h"

#include <stddef.h>

#include "geoidobject.h"

static PyObject *pyvl_line_repr(PyObject *self)
{
    const PyVL_LineObject *this = (PyVL_LineObject *)self;
    return PyUnicode_FromFormat("Line(%u, %u)", this->begin, this->end);
}

static PyObject *pyvl_line_str(PyObject *self)
{
    const PyVL_LineObject *this = (PyVL_LineObject *)self;
    return PyUnicode_FromFormat("(%u -> %u)", this->begin, this->end);
}

static PyMemberDef line_members[] = {
    {.name = "begin",
     .type = Py_T_UINT,
     .offset = offsetof(PyVL_LineObject, begin),
     .flags = 0,
     .doc = "Beginning point of the line."},
    {.name = "end",
     .type = Py_T_UINT,
     .offset = offsetof(PyVL_LineObject, end),
     .flags = 0,
     .doc = "End point of the line."},
    {},
};

PyVL_LineObject *pyvl_line_from_indices(unsigned begin, unsigned end)
{
    PyVL_LineObject *const this = (PyVL_LineObject *)pyvl_line_type.tp_alloc(&pyvl_line_type, 0);
    if (!this)
        return NULL;
    this->begin = begin;
    this->end = end;

    return this;
}

static PyObject *pyvl_line_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *a1, *a2;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char *[3]){"begin", "end", NULL}, &a1, &a2))
    {
        return NULL;
    }
    unsigned begin, end;
    if (PyObject_TypeCheck(a1, &pyvl_geoid_type))
    {
        begin = ((PyVL_GeoIDObject *)a1)->id.value;
    }
    else
    {
        begin = PyLong_AsUnsignedLong(a1);
        if (PyErr_Occurred())
            return NULL;
    }

    if (PyObject_TypeCheck(a2, &pyvl_geoid_type))
    {
        end = ((PyVL_GeoIDObject *)a2)->id.value;
    }
    else
    {
        end = PyLong_AsUnsignedLong(a2);
        if (PyErr_Occurred())
            return NULL;
    }

    PyVL_LineObject *const this = (PyVL_LineObject *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;
    this->begin = begin;
    this->end = end;

    return (PyObject *)this;
}

static PyObject *pyvl_line_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyVL_LineObject *const this = (PyVL_LineObject *)self;
    if (!PyObject_TypeCheck(other, &pyvl_line_type))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyVL_LineObject *const that = (PyVL_LineObject *)other;
    const bool val = this->begin == that->begin && this->end == that->end;
    if (op == Py_NE)
    {
        return PyBool_FromLong(!val);
    }
    return PyBool_FromLong(val);
}

PyDoc_STRVAR(pyvl_line_type_docstring, "Class which describes a connection between two points.");

PyTypeObject pyvl_line_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pyvl.cvl.Line",
    .tp_basicsize = sizeof(PyVL_LineObject),
    .tp_itemsize = 0,
    .tp_repr = pyvl_line_repr,
    .tp_str = pyvl_line_str,
    .tp_doc = pyvl_line_type_docstring,
    .tp_new = pyvl_line_new,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_members = line_members,
    .tp_richcompare = pyvl_line_rich_compare,
};
