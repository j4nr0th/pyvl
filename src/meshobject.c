//
// Created by jan on 24.11.2024.
//

#include "meshobject.h"
#include "core/mesh.h"

#include <numpy/arrayobject.h>

#include "allocator.h"
#include "lineobject.h"
#include "surfaceobject.h"

static PyObject *pydust_mesh_str(PyObject *self)
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    return PyUnicode_FromFormat("Mesh(%u points, %u lines, %u surfaces)", this->mesh.n_points, this->mesh.n_lines,
                                this->mesh.n_surfaces);
}

constexpr PyDoc_STRVAR(pydust_mesh_type_docstring, "Wrapper around cdust mesh type.");

static PyObject *pydust_mesh_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyDust_MeshObject *this = nullptr;
    real3_t *pos = nullptr;
    unsigned n_elements = 0;
    unsigned n_positions = 0;
    unsigned *per_element = nullptr;
    unsigned *flat_points = nullptr;
    PyObject *seq = nullptr;

    PyObject *positions;
    PyObject *root;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O", (char *[3]){"positions", "connectivity", nullptr},
                                     &PyArray_Type, &positions, &root))
    {
        return nullptr;
    }

    //  Load positions
    {
        PyArrayObject *real_positions = (PyArrayObject *)PyArray_FromAny(positions, PyArray_DescrFromType(NPY_DOUBLE),
                                                                         2, 2, NPY_ARRAY_C_CONTIGUOUS, nullptr);
        const npy_intp *pos_dims = PyArray_DIMS(real_positions);
        n_positions = (unsigned)pos_dims[0];
        if (pos_dims[1] != 3)
        {
            PyErr_Format(PyExc_ValueError, "Positions must be given as an (N, 3) array, but were instead (%u, %u).",
                         (unsigned)n_positions, (unsigned)pos_dims[1]);
            Py_DECREF(real_positions);
            return nullptr;
        }
        pos = PyObject_Malloc(sizeof(*pos) * n_positions);
        if (!pos)
        {
            Py_DECREF(real_positions);
            return nullptr;
        }

        memcpy(pos, PyArray_DATA(real_positions), sizeof(*pos) * n_positions);
        Py_DECREF(real_positions);
    }

    // Load element data
    {
        seq = PySequence_Fast(root, "Second parameter must be a sequence of sequences");
        if (!seq)
        {
            goto end;
        }
        n_elements = PySequence_Fast_GET_SIZE(seq);
        per_element = PyMem_Malloc(sizeof(*per_element) * n_elements);
        if (!per_element)
        {
            goto end;
        }
        unsigned total_pts = 0;
        for (unsigned i = 0; i < n_elements; ++i)
        {
            const ssize_t len = PySequence_Size(PySequence_Fast_GET_ITEM(seq, i));
            if (len < 0)
            {
                PyErr_Format(PyExc_TypeError, "Element indices for element %u were not a sequence.", i);
                goto end;
            }
            if (len < 3)
            {
                PyErr_Format(PyExc_ValueError, "Element %u had only %u indices given (at least 3 are needed).", i,
                             (unsigned)len);
                goto end;
            }
            total_pts += (unsigned)len;
            per_element[i] = (unsigned)len;
        }
        flat_points = PyMem_Malloc(sizeof(*flat_points) * total_pts);
        if (!flat_points)
        {
            goto end;
        }
        for (unsigned i = 0, j = 0; i < n_elements; ++i)
        {
            const PyArrayObject *const idx =
                (PyArrayObject *)PyArray_FromAny(PySequence_Fast_GET_ITEM(seq, i), PyArray_DescrFromType(NPY_UINT), 1,
                                                 1, NPY_ARRAY_C_CONTIGUOUS, nullptr);
            if (!idx)
            {
                goto end;
            }

            const unsigned *data = PyArray_DATA(idx);
            for (unsigned k = 0; k < per_element[i]; ++k)
            {
                const unsigned v = data[k];
                if (v > n_positions)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "Element %u had specified a point with index %u as its %u"
                                 " point, while only %u positions were given.",
                                 i, v, k, n_positions);
                    Py_DECREF(idx);
                    goto end;
                }
                flat_points[j + k] = v;
            }
            j += per_element[i];

            Py_DECREF(idx);
        }
        mesh_t *const msh = mesh_from_elements(n_elements, per_element, flat_points, &CDUST_OBJ_ALLOCATOR);
        if (!msh)
        {
            PyErr_Format(PyExc_RuntimeError, "Failed creating a mesh from given indices.");
            goto end;
        }
        this = (PyDust_MeshObject *)type->tp_alloc(type, 0);
        if (!this)
        {
            CDUST_OBJ_ALLOCATOR.deallocate(CDUST_OBJ_ALLOCATOR.state, msh);
            goto end;
        }
        this->mesh = *msh;
        // memset(msh, 0, sizeof*msh);
        CDUST_OBJ_ALLOCATOR.deallocate(CDUST_OBJ_ALLOCATOR.state, msh);
    }
    this->mesh.n_points = n_positions;
    this->mesh.positions = pos;
    pos = nullptr;

end:
    PyMem_Free(flat_points);
    PyMem_Free(per_element);
    Py_XDECREF(seq);
    PyObject_Free(pos);
    return (PyObject *)this;
}

static PyObject *pydust_mesh_get_n_points(PyObject *self, void *Py_UNUSED(closure))
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    return PyLong_FromUnsignedLong(this->mesh.n_points);
}

static PyObject *pydust_mesh_get_n_lines(PyObject *self, void *Py_UNUSED(closure))
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    return PyLong_FromUnsignedLong(this->mesh.n_lines);
}

static PyObject *pydust_mesh_get_n_surfaces(PyObject *self, void *Py_UNUSED(closure))
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    return PyLong_FromUnsignedLong(this->mesh.n_surfaces);
}

static PyObject *pydust_mesh_get_positions(PyObject *self, void *Py_UNUSED(closure))
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    const npy_intp dims[2] = {this->mesh.n_points, 3};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, this->mesh.positions);
    if (!out)
    {
        return nullptr;
    }
    if (PyArray_SetBaseObject(out, self) < 0)
    {
        Py_DECREF(out);
        return nullptr;
    }
    //  PyArray_SetBaseObject stole a reference, so increment it.
    Py_INCREF(self);
    return (PyObject *)out;
}

static int pydust_mesh_set_positions(PyObject *self, PyObject *new_value, void *Py_UNUSED(closure))
{
    PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    PyArrayObject *const in = (PyArrayObject *)PyArray_FromAny(new_value, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                               NPY_ARRAY_C_CONTIGUOUS, nullptr);
    if (!in)
    {
        return -1;
    }
    const npy_intp *in_dims = PyArray_DIMS(in);
    if (in_dims[0] != this->mesh.n_points || in_dims[1] != 3)
    {
        PyErr_Format(PyExc_ValueError,
                     "Mesh has %u points with 3d positions, so the required array shape is "
                     "(%u, 3), but an array with the shape (%u, %u) was given instead.",
                     this->mesh.n_points, this->mesh.n_points, (unsigned)in_dims[0], (unsigned)in_dims[1]);
        Py_DECREF(in);
        return -1;
    }
    memcpy(this->mesh.positions, PyArray_DATA(in), sizeof(*this->mesh.positions) * this->mesh.n_points);

    Py_DECREF(in);
    return 0;
}

static PyGetSetDef pydust_mesh_getset[] = {
    {.name = "n_points",
     .get = pydust_mesh_get_n_points,
     .set = nullptr,
     .doc = "Number of points in the mesh",
     .closure = nullptr},
    {.name = "n_lines",
     .get = pydust_mesh_get_n_lines,
     .set = nullptr,
     .doc = "Number of lines in the mesh",
     .closure = nullptr},
    {.name = "n_surfaces",
     .get = pydust_mesh_get_n_surfaces,
     .set = nullptr,
     .doc = "Number of surfaces in the mesh",
     .closure = nullptr},
    {.name = "positions",
     .get = pydust_mesh_get_positions,
     .set = pydust_mesh_set_positions,
     .doc = "Positions of mesh points.",
     .closure = nullptr},
    {},
};

static PyObject *pydust_mesh_get_line(PyObject *self, PyObject *arg)
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    long idx = PyLong_AsLong(arg);
    if (PyErr_Occurred())
    {
        return nullptr;
    }
    if (idx >= this->mesh.n_lines || idx < -(long)this->mesh.n_lines)
    {
        PyErr_Format(PyExc_IndexError, "Index %ld is our of bounds for a mesh with %u lines.", idx, this->mesh.n_lines);
        return nullptr;
    }
    unsigned i;
    if (idx < 0)
    {
        i = (unsigned)((long)this->mesh.n_lines + idx);
    }
    else
    {
        i = (unsigned)idx;
    }
    return (PyObject *)pydust_line_from_indices(this->mesh.lines[i].p1.value, this->mesh.lines[i].p2.value);
}

static PyObject *pydust_mesh_get_surface(PyObject *self, PyObject *arg)
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    long idx = PyLong_AsLong(arg);
    if (PyErr_Occurred())
    {
        return nullptr;
    }
    if (idx >= this->mesh.n_surfaces || idx < -(long)this->mesh.n_surfaces)
    {
        PyErr_Format(PyExc_IndexError, "Index %ld is our of bounds for a mesh with %u surfaces.", idx,
                     this->mesh.n_surfaces);
        return nullptr;
    }
    unsigned i;
    if (idx < 0)
    {
        i = (unsigned)((long)this->mesh.n_surfaces + idx);
    }
    else
    {
        i = (unsigned)idx;
    }

    return (PyObject *)pydust_surface_from_mesh_surface(&this->mesh, (geo_id_t){.orientation = 0, .value = idx});
}

static PyObject *pydust_mesh_compute_dual(PyObject *self, PyObject *Py_UNUSED(arg))
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    mesh_t *dual = mesh_dual_from_primal(&this->mesh, &CDUST_OBJ_ALLOCATOR);
    if (!dual)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not compute dual to the mesh.");
        return nullptr;
    }
    PyDust_MeshObject *that = (PyDust_MeshObject *)pydust_mesh_type.tp_alloc(&pydust_mesh_type, 0);
    if (!that)
    {
        mesh_free(dual, &CDUST_OBJ_ALLOCATOR);
        return nullptr;
    }
    memcpy(&that->mesh, dual, sizeof(that->mesh));
    PyObject_Free(dual);
    return (PyObject *)that;
}

static void cleanup_memory(PyObject *cap)
{
    void *const ptr = PyCapsule_GetPointer(cap, nullptr);
    PyMem_Free(ptr);
}

static PyObject *pydust_mesh_to_element_connectivity(PyObject *self, PyObject *Py_UNUSED(arg))
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    unsigned *point_counts, *flat_points;
    const unsigned n_elements = mesh_to_elements(&this->mesh, &point_counts, &flat_points, &CDUST_MEM_ALLOCATOR);
    if (n_elements != this->mesh.n_surfaces)
    {
        if (!PyErr_Occurred())
        {
            PyErr_Format(PyExc_RuntimeError, "Could not convert mesh to elements.");
        }
        return nullptr;
    }
    PyObject *const cap = PyCapsule_New(point_counts, nullptr, cleanup_memory);
    if (!cap)
    {
        PyMem_Free(point_counts);
        PyMem_Free(flat_points);
        return nullptr;
    }
    const npy_intp n_counts = n_elements;
    PyObject *const counts_array = PyArray_SimpleNewFromData(1, &n_counts, NPY_UINT, point_counts);
    if (!counts_array)
    {
        Py_DECREF(cap);
        PyMem_Free(flat_points);
        return nullptr;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)counts_array, cap) < 0)
    {
        Py_DECREF(counts_array);
        Py_DECREF(cap);
        PyMem_Free(flat_points);
        return nullptr;
    }

    PyObject *const cap_2 = PyCapsule_New(flat_points, nullptr, cleanup_memory);
    if (!cap_2)
    {
        Py_DECREF(counts_array);
        PyMem_Free(flat_points);
        return nullptr;
    }
    npy_intp n_flat = 0;
    for (unsigned i = 0; i < n_elements; ++i)
    {
        n_flat += point_counts[i];
    }
    PyObject *const points_array = PyArray_SimpleNewFromData(1, &n_flat, NPY_UINT, flat_points);
    if (!points_array)
    {
        Py_DECREF(counts_array);
        Py_DECREF(cap_2);
        return nullptr;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)points_array, cap_2) < 0)
    {
        Py_DECREF(counts_array);
        Py_DECREF(points_array);
        return nullptr;
    }

    PyObject *out = PyTuple_Pack(2, counts_array, points_array);
    if (!out)
    {
        Py_DECREF(counts_array);
        Py_DECREF(points_array);
    }
    return out;
}

static PyMethodDef pydust_mesh_methods[] = {
    {.ml_name = "get_line",
     .ml_meth = pydust_mesh_get_line,
     .ml_flags = METH_O,
     .ml_doc = "Get the line from the mesh."},
    {.ml_name = "get_surface",
     .ml_meth = pydust_mesh_get_surface,
     .ml_flags = METH_O,
     .ml_doc = "Get the surface from the mesh."},
    {.ml_name = "compute_dual",
     .ml_meth = pydust_mesh_compute_dual,
     .ml_flags = METH_NOARGS,
     .ml_doc = "Create dual to the mesh."},
    {.ml_name = "to_element_connectivity",
     .ml_meth = pydust_mesh_to_element_connectivity,
     .ml_flags = METH_NOARGS,
     .ml_doc = "Convert mesh connectivity to arrays list of element lengths and indices."},
    {},
};

CDUST_INTERNAL
PyTypeObject pydust_mesh_type = {
    .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "cdust.Mesh",
    .tp_basicsize = sizeof(PyDust_MeshObject),
    .tp_itemsize = 0,
    .tp_str = pydust_mesh_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .tp_doc = pydust_mesh_type_docstring,
    .tp_methods = pydust_mesh_methods,
    .tp_getset = pydust_mesh_getset,
    .tp_new = pydust_mesh_new,
};
