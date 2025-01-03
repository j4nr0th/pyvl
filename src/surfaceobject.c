//
// Created by jan on 24.11.2024.
//

#include "surfaceobject.h"

#include "lineobject.h"

static PyObject *pyvl_surface_repr(PyObject *self)
{
    const PyVL_SurfaceObject *this = (PyVL_SurfaceObject *)self;
    size_t len = 0;
    len += snprintf(NULL, 0, "Surface((");
    for (unsigned i = 0; i < this->n_lines; ++i)
    {
        len += snprintf(NULL, 0, " Line(%u, %u),", this->lines[i].p1.value, this->lines[i].p2.value);
    }
    len += snprintf(NULL, 0, "))");
    char *const buffer = PyMem_Malloc(len + 1);
    size_t used = 0;
    if (!buffer)
    {
        return NULL;
    }
    used += snprintf(buffer + used, len - used + 1, "Surface((");
    for (unsigned i = 0; i < this->n_lines; ++i)
    {
        used +=
            snprintf(buffer + used, len - used + 1, " Line(%u, %u),", this->lines[i].p1.value, this->lines[i].p2.value);
    }
    snprintf(buffer + used, len - used + 1, "))");
    PyObject *ret = PyUnicode_FromString(buffer);
    PyMem_Free(buffer);
    return ret;
}

static PyObject *pyvl_surface_str(PyObject *self)
{
    const PyVL_SurfaceObject *this = (PyVL_SurfaceObject *)self;
    size_t len = 0;
    len += snprintf(NULL, 0, "(");
    for (unsigned i = 0; i < this->n_lines - 1; ++i)
    {
        len += snprintf(NULL, 0, "(%u -> %u) -> ", this->lines[i].p1.value, this->lines[i].p2.value);
    }
    if (this->n_lines != 0)
    {
        len += snprintf(NULL, 0, "(%u -> %u)", this->lines[this->n_lines - 1].p1.value,
                        this->lines[this->n_lines - 1].p2.value);
    }
    len += snprintf(NULL, 0, ")");
    char *const buffer = PyMem_Malloc(len + 1);
    size_t used = 0;
    if (!buffer)
    {
        return NULL;
    }
    used += snprintf(buffer + used, len - used + 1, "(");
    for (unsigned i = 0; i < this->n_lines - 1; ++i)
    {
        used +=
            snprintf(buffer + used, len - used + 1, "(%u -> %u) -> ", this->lines[i].p1.value, this->lines[i].p2.value);
    }
    if (this->n_lines != 0)
    {
        used += snprintf(buffer + used, len - used + 1, "(%u -> %u)", this->lines[this->n_lines - 1].p1.value,
                         this->lines[this->n_lines - 1].p2.value);
    }
    snprintf(buffer + used, len - used + 1, ")");
    PyObject *ret = PyUnicode_FromString(buffer);
    PyMem_Free(buffer);
    return ret;
}

PyDoc_STRVAR(pyvl_surface_type_docstring, "Surface bound by a set of lines.");

static PyObject *pyvl_surface_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyVL_SurfaceObject *const this = (PyVL_SurfaceObject *)self;
    if (!PyObject_TypeCheck(other, &pyvl_surface_type))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const PyVL_SurfaceObject *const that = (PyVL_SurfaceObject *)other;
    bool val = false;
    if (this->n_lines != that->n_lines)
        goto end;

    unsigned offset;
    for (offset = 0; offset < this->n_lines; ++offset)
    {
        if (this->lines[0].p1.value == that->lines[offset].p1.value &&
            this->lines[0].p2.value == that->lines[offset].p2.value)
            break;
    }
    if (offset == this->n_lines)
        goto end;

    for (unsigned i = 1; i < this->n_lines; ++i)
    {
        const unsigned j = (i + offset) % this->n_lines;
        if (this->lines[i].p1.value != that->lines[j].p1.value || this->lines[i].p2.value != that->lines[j].p2.value)
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

static PyObject *pyvl_surface_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *arg;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char *[2]){"lines", NULL}, &arg))
    {
        return NULL;
    }

    PyObject *const seq = PySequence_Fast(arg, "Argument was not a sequence.");
    if (!seq)
        return NULL;
    const unsigned len = PySequence_Fast_GET_SIZE(seq);
    for (unsigned i = 0; i < len; ++i)
    {
        const PyObject *const v = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyObject_TypeCheck(v, &pyvl_line_type))
        {
            PyErr_Format(PyExc_TypeError, "Element at index %u (%R) is not a Line object.", i, v);
            goto failed;
        }
    }
    const PyVL_LineObject *const v0 = (PyVL_LineObject *)PySequence_Fast_GET_ITEM(seq, 0);
    const unsigned first = v0->begin;
    unsigned last = v0->end;
    for (unsigned i = 1; i < len; ++i)
    {
        const PyVL_LineObject *const v = (PyVL_LineObject *)PySequence_Fast_GET_ITEM(seq, i);
        if (v->begin != last)
        {
            PyErr_Format(PyExc_ValueError, "Line %u begins at point %u, but previous line begins with point %u.", i,
                         v->begin, last);
            goto failed;
        }
        last = v->end;
    }
    if (first != last)
    {
        PyErr_Format(PyExc_ValueError, "First line begins at point %u, but last line ends at point %u", first, last);
        goto failed;
    }

    PyVL_SurfaceObject *const this = (PyVL_SurfaceObject *)type->tp_alloc(type, len);
    if (!this)
        goto failed;
    this->n_lines = len;
    for (unsigned i = 0; i < len; ++i)
    {
        const PyVL_LineObject *const v = (PyVL_LineObject *)PySequence_Fast_GET_ITEM(seq, i);
        this->lines[i] =
            (line_t){.p1 = {.orientation = 0, .value = v->begin}, .p2 = {.orientation = 0, .value = v->end}};
    }

    Py_DECREF(seq);
    return (PyObject *)this;
failed:
    Py_DECREF(seq);
    return NULL;
}

CVL_INTERNAL
PyVL_SurfaceObject *pyvl_surface_from_points(unsigned n_points,
                                             const unsigned CVL_ARRAY_ARG(points, static restrict n_points))
{
    PyVL_SurfaceObject *const this =
        (PyVL_SurfaceObject *)pyvl_surface_type.tp_alloc(&pyvl_surface_type, (Py_ssize_t)n_points);
    if (!this)
        return NULL;
    this->n_lines = n_points;
    for (unsigned i = 0; i < n_points - 1; ++i)
    {
        this->lines[i] =
            (line_t){.p1 = {.orientation = 0, .value = points[i]}, .p2 = {.orientation = 0, .value = points[i + 1]}};
    }
    this->lines[n_points - 1] =
        (line_t){.p1 = {.orientation = 0, .value = points[n_points - 1]}, .p2 = {.orientation = 0, .value = points[0]}};
    return this;
}

CVL_INTERNAL
PyVL_SurfaceObject *pyvl_surface_from_lines(unsigned n, const line_t CVL_ARRAY_ARG(lines, static restrict n))
{
    PyVL_SurfaceObject *const this =
        (PyVL_SurfaceObject *)pyvl_surface_type.tp_alloc(&pyvl_surface_type, (Py_ssize_t)n);
    if (!this)
        return NULL;
    this->n_lines = n;
    for (unsigned i = 0; i < n; ++i)
    {
        this->lines[i] = lines[i];
    }
    return this;
}

CVL_INTERNAL
PyVL_SurfaceObject *pyvl_surface_from_mesh_surface(const mesh_t *msh, geo_id_t id)
{
    const unsigned idx = id.value;
    const unsigned i0 = msh->surface_offsets[idx], i1 = msh->surface_offsets[idx + 1];
    PyVL_SurfaceObject *const this =
        (PyVL_SurfaceObject *)pyvl_surface_type.tp_alloc(&pyvl_surface_type, (Py_ssize_t)(i1 - i0));
    if (!this)
        return NULL;
    for (unsigned i = 0; i < i1 - i0; ++i)
    {
        const geo_id_t lid = msh->surface_lines[i + i0];
        if (lid.orientation ^ id.orientation)
        {
            this->lines[i] = msh->lines[lid.value];
        }
        else
        {
            this->lines[i] = (line_t){.p1 = msh->lines[lid.value].p2, .p2 = msh->lines[lid.value].p1};
        }
    }
    return this;
}

static PyObject *pyvl_surface_get_lines(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_SurfaceObject *this = (PyVL_SurfaceObject *)self;
    PyObject *const out = PyTuple_New(this->n_lines);
    for (unsigned i = 0; i < this->n_lines; ++i)
    {
        PyVL_LineObject *ln = pyvl_line_from_indices(this->lines[i].p1.value, this->lines[i].p2.value);
        if (!ln)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, (PyObject *)ln);
    }

    return out;
}

static PyObject *pyvl_surface_get_n_lines(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_SurfaceObject *this = (PyVL_SurfaceObject *)self;
    return PyLong_FromUnsignedLong(this->n_lines);
}

static PyGetSetDef pyvl_surface_getset[] = {
    {.name = "lines",
     .get = pyvl_surface_get_lines,
     .set = NULL,
     .doc = "Lines that make up the surface.",
     .closure = NULL},
    {.name = "n_lines",
     .get = pyvl_surface_get_n_lines,
     .set = NULL,
     .doc = "Number of the lines that make up the surface.",
     .closure = NULL},
    {0},
};

CVL_INTERNAL
PyTypeObject pyvl_surface_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pyvl.cvl.Surface",
    .tp_basicsize = sizeof(PyVL_SurfaceObject),
    .tp_itemsize = sizeof(line_t),
    .tp_repr = pyvl_surface_repr,
    .tp_str = pyvl_surface_str,
    .tp_doc = pyvl_surface_type_docstring,
    .tp_richcompare = pyvl_surface_rich_compare,
    .tp_getset = pyvl_surface_getset,
    .tp_new = pyvl_surface_new,
};
