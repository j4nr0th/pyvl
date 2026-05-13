#include <cpyutl.h>

#include "geoidobject.h"
#include "lineobject.h"
#include <structmember.h>

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
    {
        .name = "begin",
        .type = T_UINT,
        .offset = offsetof(PyVL_LineObject, begin),
        .flags = 0,
        .doc = "Beginning point of the line.",
    },
    {
        .name = "end",
        .type = T_UINT,
        .offset = offsetof(PyVL_LineObject, end),
        .flags = 0,
        .doc = "End point of the line.",
    },
    {0},
};

PyVL_LineObject *pyvl_line_from_indices(PyTypeObject *line_type, const unsigned begin, const unsigned end)
{
    PyVL_LineObject *const this = (PyVL_LineObject *)line_type->tp_alloc(line_type, 0);
    if (!this)
        return NULL;
    this->begin = begin;
    this->end = end;

    return this;
}

static PyObject *pyvl_line_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    const module_state_t *const state = get_module_state(type);
    if (!state)
        return NULL;

    PyObject *a1, *a2;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char *[3]){"begin", "end", NULL}, &a1, &a2))
    {
        return NULL;
    }
    unsigned begin, end;
    if (PyObject_TypeCheck(a1, state->geoid_type))
    {
        begin = ((PyVL_GeoIDObject *)a1)->id.value;
    }
    else
    {
        begin = PyLong_AsUnsignedLong(a1);
        if (PyErr_Occurred())
            return NULL;
    }

    if (PyObject_TypeCheck(a2, state->geoid_type))
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
    const module_state_t *const state = get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;

    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyVL_LineObject *const this = (PyVL_LineObject *)self;
    if (!PyObject_TypeCheck(other, state->line_type))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyVL_LineObject *const that = (PyVL_LineObject *)other;
    const bool val = (this->begin == that->begin && this->end == that->end) != 0;
    if (op == Py_NE)
    {
        return PyBool_FromLong(!val);
    }
    return PyBool_FromLong((long)val);
}

PyDoc_STRVAR(pyvl_line_type_docstring, "Class which describes a connection between two points.");

PyType_Spec pyvl_line_typespec = {
    .name = PYVL_CTYPE_NAME(Line),
    .basicsize = sizeof(PyVL_LineObject),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC,
    .slots =
        (PyType_Slot[]){
            {Py_tp_repr, pyvl_line_repr},
            {Py_tp_str, pyvl_line_str},
            {Py_tp_doc, (void *)pyvl_line_type_docstring},
            {Py_tp_new, pyvl_line_new},
            {Py_tp_members, line_members},
            {Py_tp_richcompare, pyvl_line_rich_compare},
            {Py_tp_traverse, cpyutl_traverse_heap_type},
            {0}, // Sentinel
        },
};
