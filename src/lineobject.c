//
// Created by jan on 23.11.2024.
//

#include "lineobject.h"

#include <stddef.h>

#include "geoidobject.h"

static PyObject *pydust_line_repr(PyObject *self)
{
    const PyDust_LineObject *this = (PyDust_LineObject *)self;
    return PyUnicode_FromFormat("Line(%u, %u)", this->begin, this->end);
}

static PyObject *pydust_line_str(PyObject *self)
{
    const PyDust_LineObject *this = (PyDust_LineObject *)self;
    return PyUnicode_FromFormat("(%u -> %u)", this->begin, this->end);
}

static PyMemberDef line_members[] =
    {
        {.name = "begin", .type = Py_T_UINT, .offset = offsetof(PyDust_LineObject, begin), .flags = 0, .doc = "Beginning point of the line."},
        {.name = "end", .type = Py_T_UINT, .offset = offsetof(PyDust_LineObject, end), .flags = 0, .doc = "End point of the line."},
        {},
    };


PyDust_LineObject* pydust_line_from_indices(unsigned begin, unsigned end)
{
    PyDust_LineObject *const this = (PyDust_LineObject *)pydust_line_type.tp_alloc(&pydust_line_type, 0);
    if (!this) return nullptr;
    this->begin = begin;
    this->end = end;

    return this;
}

static PyObject *pydust_line_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *a1, *a2;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char*[3]){"begin", "end", nullptr}, &a1, &a2))
    {
        return nullptr;
    }
    unsigned begin, end;
    if (PyObject_TypeCheck(a1, &pydust_geoid_type))
    {
        begin = ((PyDust_GeoIDObject *)a1)->id.value;
    }
    else
    {
        begin = PyLong_AsUnsignedLong(a1);
        if (PyErr_Occurred()) return nullptr;
    }

    if (PyObject_TypeCheck(a2, &pydust_geoid_type))
    {
        end = ((PyDust_GeoIDObject *)a2)->id.value;
    }
    else
    {
        end = PyLong_AsUnsignedLong(a2);
        if (PyErr_Occurred()) return nullptr;
    }

    PyDust_LineObject *const this = (PyDust_LineObject *)type->tp_alloc(type, 0);
    if (!this) return nullptr;
    this->begin = begin;
    this->end = end;

    return (PyObject *)this;
}

static PyObject *pydust_line_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyDust_LineObject *const this = (PyDust_LineObject *)self;
    if (!PyObject_TypeCheck(other, &pydust_line_type))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyDust_LineObject *const that = (PyDust_LineObject *)other;
    const bool val = this->begin == that->begin && this->end == that->end;
    if (op == Py_NE)
    {
        return PyBool_FromLong(!val);
    }
    return PyBool_FromLong(val);
}

constexpr
PyDoc_STRVAR(pydust_line_type_docstring, "Class which describes a connection between two points.");

PyTypeObject pydust_line_type =
    {
    .ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "cdust.Line",
    .tp_basicsize = sizeof(PyDust_LineObject),
    .tp_itemsize = 0,
    .tp_repr = pydust_line_repr,
    .tp_str = pydust_line_str,
    .tp_doc = pydust_line_type_docstring,
    .tp_new = pydust_line_new,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_IMMUTABLETYPE,
    .tp_members = line_members,
    .tp_richcompare = pydust_line_rich_compare,
    };

