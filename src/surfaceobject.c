//
// Created by jan on 24.11.2024.
//

#include "surfaceobject.h"

#include "lineobject.h"

static PyObject *pydust_surface_repr(PyObject *self)
{
    const PyDust_SurfaceObject *this = (PyDust_SurfaceObject *)self;
    size_t len = 0;
    len += snprintf(nullptr, 0, "Surface((");
    for (unsigned i = 0; i < this->n_lines; ++i)
    {
        len += snprintf(nullptr, 0, " Line(%u, %u),", this->lines[i].p1.value, this->lines[i].p2.value);
    }
    len += snprintf(nullptr, 0, "))");
    char *const buffer = PyMem_Malloc(len + 1);
    size_t used = 0;
    if (!buffer)
    {
        return nullptr;
    }
    used += snprintf(buffer + used, len - used + 1, "Surface((");
    for (unsigned i = 0; i < this->n_lines; ++i)
    {
        used += snprintf(buffer + used, len - used + 1, " Line(%u, %u),", this->lines[i].p1.value, this->lines[i].p2.value);
    }
    snprintf(buffer + used, len - used + 1, "))");
    PyObject *ret = PyUnicode_FromString(buffer);
    PyMem_Free(buffer);
    return ret;
}

static PyObject *pydust_surface_str(PyObject *self)
{
    const PyDust_SurfaceObject *this = (PyDust_SurfaceObject *)self;
    size_t len = 0;
    len += snprintf(nullptr, 0, "(");
    for (unsigned i = 0; i < this->n_lines - 1; ++i)
    {
        len += snprintf(nullptr, 0, "(%u -> %u) -> ", this->lines[i].p1.value, this->lines[i].p2.value);
    }
    if (this->n_lines != 0)
    {
        len += snprintf(nullptr, 0, "(%u -> %u)", this->lines[this->n_lines - 1].p1.value, this->lines[this->n_lines - 1].p2.value);
    }
    len += snprintf(nullptr, 0, ")");
    char *const buffer = PyMem_Malloc(len + 1);
    size_t used = 0;
    if (!buffer)
    {
        return nullptr;
    }
    used += snprintf(buffer + used, len - used + 1, "(");
    for (unsigned i = 0; i < this->n_lines - 1; ++i)
    {
        used += snprintf(buffer + used, len - used + 1, "(%u -> %u) -> ", this->lines[i].p1.value, this->lines[i].p2.value);
    }
    if (this->n_lines != 0)
    {
        used += snprintf(buffer + used, len - used + 1, "(%u -> %u)", this->lines[this->n_lines - 1].p1.value, this->lines[this->n_lines - 1].p2.value);
    }
    snprintf(buffer + used, len - used + 1, ")");
    PyObject *ret = PyUnicode_FromString(buffer);
    PyMem_Free(buffer);
    return ret;
}

constexpr
PyDoc_STRVAR(pydust_surface_type_docstring, "Surface bound by a set of lines.");

static PyObject *pydust_surface_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyDust_SurfaceObject *const this = (PyDust_SurfaceObject *)self;
    if (!PyObject_TypeCheck(other, &pydust_surface_type))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyDust_SurfaceObject *const that = (PyDust_SurfaceObject *)other;
    bool val = false;
    if (this->n_lines != that->n_lines) goto end;

    unsigned offset;
    for (offset = 0; offset < this->n_lines; ++offset)
    {
        if (this->lines[0].p1.value == that->lines[offset].p1.value
            && this->lines[0].p2.value == that->lines[offset].p2.value) break;
    }
    if (offset == this->n_lines) goto end;

    for (unsigned i = 1; i < this->n_lines; ++i)
    {
        const unsigned j = (i + offset) % this->n_lines;
        if (this->lines[i].p1.value != that->lines[j].p1.value
            || this->lines[i].p2.value != that->lines[j].p2.value)
        {
            goto end;
        }
    }
    val = true;

end:
    if (op == Py_NE)
    {
        return PyBool_FromLong(!val);
    }
    return PyBool_FromLong(val);
}

static PyObject *pydust_surface_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *arg;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char*[2]){"lines", nullptr}, &arg))
    {
        return nullptr;
    }

    PyObject *const seq = PySequence_Fast(arg, "Argument was not a sequence.");
    if (!seq) return nullptr;
    const unsigned len = PySequence_Fast_GET_SIZE(seq);
    for (unsigned i = 0; i < len; ++i)
    {
        const PyObject *const v = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyObject_TypeCheck(v, &pydust_line_type))
        {
            PyErr_Format(PyExc_TypeError, "Element at index %u (%R) is not a Line object.", i, v);
            goto failed;
        }
    }
    const PyDust_LineObject *const v0 = (PyDust_LineObject *)PySequence_Fast_GET_ITEM(seq, 0);
    const unsigned first = v0->begin;
    unsigned last = v0->end;
    for (unsigned i = 1; i < len; ++i)
    {
        const PyDust_LineObject *const v = (PyDust_LineObject *)PySequence_Fast_GET_ITEM(seq, i);
        if (v->begin != last)
        {
            PyErr_Format(PyExc_TypeError, "Line %u begins at point %u, but previous line begins with point %u.", i, v->begin, last);
            goto failed;
        }
        last = v->end;
    }
    if (first != last)
    {
        PyErr_Format(PyExc_TypeError, "First line begins at point %u, but last line ends at point %u", first, last);
        goto failed;
    }

    PyDust_SurfaceObject *const this = (PyDust_SurfaceObject *)type->tp_alloc(type, len);
    if (!this) goto failed;
    this->n_lines = len;
    for (unsigned i = 0; i < len; ++i)
    {
        const PyDust_LineObject *const v = (PyDust_LineObject *)PySequence_Fast_GET_ITEM(seq, i);
        this->lines[i] =
            (line_t){.p1 = {.orientation = 0, .value = v->begin}, .p2 = {.orientation = 0, .value = v->end}};
    }

    Py_DECREF(seq);
    return (PyObject *)this;
failed:
    Py_DECREF(seq);
    return nullptr;
}

static PyObject *pydust_surface_get_lines(PyObject *self, void *Py_UNUSED(closure))
{
    const PyDust_SurfaceObject *this = (PyDust_SurfaceObject *)self;
    PyObject *const out = PyTuple_New(this->n_lines);
    for (unsigned i = 0; i < this->n_lines; ++i)
    {
        PyDust_LineObject *ln = pydust_line_from_indices(this->lines[i].p1.value, this->lines[i].p2.value);
        if (!ln)
        {
            Py_DECREF(out);
            return nullptr;
        }
        PyTuple_SET_ITEM(out, i, ln);
    }

    return out;
}

static PyObject *pydust_surface_get_n_lines(PyObject *self, void *Py_UNUSED(closure))
{
    const PyDust_SurfaceObject *this = (PyDust_SurfaceObject *)self;
    return PyLong_FromUnsignedLong(this->n_lines);
}


static PyGetSetDef pydust_surface_getset[] = {
    {.name = "lines", .get = pydust_surface_get_lines, .set = nullptr, .doc = "Lines that make up the surface.", .closure = nullptr},
    {.name = "n_lines", .get = pydust_surface_get_n_lines, .set = nullptr, .doc = "Number of the lines that make up the surface.", .closure = nullptr},
    {},    
};

CDUST_INTERNAL
PyTypeObject pydust_surface_type =
    {
    .ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "cdust.Surface",
    .tp_basicsize = sizeof(PyDust_SurfaceObject),
    .tp_itemsize = sizeof(geo_id_t),
    .tp_repr = pydust_surface_repr,
    .tp_str = pydust_surface_str,
    .tp_doc = pydust_surface_type_docstring,
    .tp_richcompare = pydust_surface_rich_compare,
    .tp_getset = pydust_surface_getset,
    .tp_new = pydust_surface_new,
    };

