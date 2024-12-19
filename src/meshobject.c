//
// Created by jan on 24.11.2024.
//

#include "meshobject.h"
#include "core/flow_solver.h"
#include "core/mesh.h"

#include <numpy/arrayobject.h>

#include "allocator.h"
#include "common.h"
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

static void pydust_mesh_dealloc(PyObject *self)
{
    PyDust_MeshObject *this = (PyDust_MeshObject *)self;

    CDUST_OBJ_ALLOCATOR.deallocate(CDUST_OBJ_ALLOCATOR.state, this->mesh.positions);
    CDUST_OBJ_ALLOCATOR.deallocate(CDUST_OBJ_ALLOCATOR.state, this->mesh.lines);
    CDUST_OBJ_ALLOCATOR.deallocate(CDUST_OBJ_ALLOCATOR.state, this->mesh.surfaces);

    Py_TYPE(this)->tp_free(this);
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

static PyObject *pydust_mesh_get_surface_normals(PyObject *self, void *Py_UNUSED(closure))
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    const npy_intp dims[2] = {this->mesh.n_surfaces, 3};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (!out)
    {
        return nullptr;
    }

    _Static_assert(sizeof(npy_float64) * 3 == sizeof(real3_t), "Binary compatibility");

    real3_t *const p_out = PyArray_DATA(out);
    if (!p_out)
    {
        Py_DECREF(out);
        return nullptr;
    }

    for (unsigned i_surf = 0; i_surf < this->mesh.n_surfaces; ++i_surf)
    {
        p_out[i_surf] = surface_normal(&this->mesh, (geo_id_t){.orientation = 0, .value = i_surf});
    }

    return (PyObject *)out;
}

static PyObject *pydust_mesh_get_surface_centers(PyObject *self, void *Py_UNUSED(closure))
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    const npy_intp dims[2] = {this->mesh.n_surfaces, 3};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (!out)
    {
        return nullptr;
    }

    _Static_assert(sizeof(npy_float64) * 3 == sizeof(real3_t), "Binary compatibility");

    real3_t *const p_out = PyArray_DATA(out);
    if (!p_out)
    {
        Py_DECREF(out);
        return nullptr;
    }

    for (unsigned i_surf = 0; i_surf < this->mesh.n_surfaces; ++i_surf)
    {
        p_out[i_surf] = surface_center(&this->mesh, (geo_id_t){.orientation = 0, .value = i_surf});
    }

    return (PyObject *)out;
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
    {.name = "surface_normals",
     .get = pydust_mesh_get_surface_normals,
     .set = nullptr,
     .doc = "Compute normals to each surface of the mesh.",
     .closure = nullptr},
    {.name = "surface_centers",
     .get = pydust_mesh_get_surface_centers,
     .set = nullptr,
     .doc = "Compute centers of each surface element.",
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

static real3_t *ensure_line_memory(PyObject *in, unsigned n_lines, unsigned n_cpts)
{
    if (!PyArray_Check(in))
    {
        PyErr_SetString(PyExc_TypeError, "Line computation buffer is not an array.");
        return nullptr;
    }
    PyArrayObject *const this = (PyArrayObject *)in;
    if (PyArray_TYPE(this) != NPY_FLOAT64)
    {
        PyErr_SetString(PyExc_ValueError, "Line computation buffer was not an array of numpy.float64.");
        return nullptr;
    }

    if (!PyArray_CHKFLAGS(this, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE))
    {
        PyErr_SetString(PyExc_ValueError, "Line computation buffer was not writable, C-contiguous, and aligned.");
        return nullptr;
    }

    if (PyArray_SIZE(this) < n_lines * n_cpts * 3)
    {
        PyErr_Format(PyExc_ValueError,
                     "Line computation buffer did not have space for enough elements "
                     "(required %zu, but got %zu).",
                     (size_t)(n_lines)*n_cpts * 3, (size_t)PyArray_SIZE(this));
        return nullptr;
    }
    return PyArray_DATA(this);
}

static PyObject *pydust_mesh_induction_matrix3(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    if (nargs != 5 && nargs != 4 && nargs != 3)
    {
        PyErr_Format(PyExc_TypeError, "Method requires 3, 4, or 5 arguments, but was called with %u.", (unsigned)nargs);
        return nullptr;
    }
    const double tol = PyFloat_AsDouble(args[0]);
    if (PyErr_Occurred())
        return nullptr;

    PyArrayObject *const in_array =
        pydust_ensure_array(args[1], 2, (const npy_intp[2]){0, 3}, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                            NPY_FLOAT64, "Control point array");
    if (!in_array)
        return nullptr;
    const npy_intp ndim = PyArray_NDIM(in_array);
    const npy_intp *dims = PyArray_DIMS(in_array);
    const unsigned n_cpts = dims[0];

    PyArrayObject *const norm_array = pydust_ensure_array(
        args[2], ndim, dims, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Normal array");
    if (!norm_array)
        return nullptr;

    PyArrayObject *out_array;
    if (nargs > 3 && !Py_IsNone(args[3]))
    {
        // If None is second arg, treat it as if it is not present at all.
        out_array = pydust_ensure_array(args[3], 2, (const npy_intp[3]){n_cpts, this->mesh.n_surfaces},
                                        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED, NPY_FLOAT64,
                                        "Output array");
        Py_INCREF(out_array);
    }
    else
    {
        const npy_intp out_dims[2] = {n_cpts, this->mesh.n_surfaces};
        out_array = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_FLOAT64);
        if (!out_array)
            return nullptr;
    }

    bool free_mem;
    real3_t *line_buffer;
    if (nargs == 5 && !Py_IsNone(args[4]))
    {
        line_buffer = ensure_line_memory(args[4], this->mesh.n_lines, n_cpts);
        if (!line_buffer)
        {
            Py_DECREF(out_array);
            return nullptr;
        }
        free_mem = false;
    }
    else
    {
        line_buffer = PyMem_Malloc(sizeof(*line_buffer) * this->mesh.n_lines * n_cpts);
        if (!line_buffer)
        {
            Py_DECREF(out_array);
            return nullptr;
        }
        free_mem = true;
    }

    // Now I can be sure the arrays are well-behaved
    const real3_t *restrict control_pts = PyArray_DATA(in_array);
    const real3_t *restrict normals = PyArray_DATA(norm_array);
    real_t *restrict out_ptr = PyArray_DATA(out_array);

    compute_line_induction(this->mesh.n_lines, this->mesh.lines, this->mesh.n_points, this->mesh.positions, n_cpts,
                           control_pts, line_buffer, tol);
    line_induction_to_normal_surface_induction(this->mesh.n_surfaces, this->mesh.surfaces, this->mesh.n_lines, n_cpts,
                                               normals, line_buffer, out_ptr);

    if (free_mem)
        PyMem_Free(line_buffer);

    return (PyObject *)out_array;
}

static PyObject *pydust_mesh_induction_matrix(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    if (nargs != 4 && nargs != 3 && nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Method requires 2, 3, or 4 arguments, but was called with %u.", (unsigned)nargs);
        return nullptr;
    }
    const double tol = PyFloat_AsDouble(args[0]);
    if (PyErr_Occurred())
        return nullptr;

    PyArrayObject *const in_array =
        pydust_ensure_array(args[1], 2, (const npy_intp[2]){0, 3}, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                            NPY_FLOAT64, "Control point array");
    if (!in_array)
        return nullptr;
    const npy_intp *dims = PyArray_DIMS(in_array);
    const unsigned n_cpts = dims[0];

    PyArrayObject *out_array;
    if (nargs > 2 && !Py_IsNone(args[2]))
    {
        // If None is second arg, treat it as if it is not present at all.
        out_array = pydust_ensure_array(args[2], 3, (const npy_intp[3]){n_cpts, this->mesh.n_surfaces, 3},
                                        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, NPY_FLOAT64,
                                        "Output tensor");
        if (!out_array)
            return nullptr;
        Py_INCREF(out_array);
    }
    else
    {
        const npy_intp out_dims[3] = {n_cpts, this->mesh.n_surfaces, 3};
        out_array = (PyArrayObject *)PyArray_SimpleNew(3, out_dims, NPY_FLOAT64);
        if (!out_array)
            return nullptr;
    }

    bool free_mem;
    real3_t *line_buffer;
    if (nargs == 4 && !Py_IsNone(args[3]))
    {
        line_buffer = ensure_line_memory(args[3], this->mesh.n_lines, n_cpts);
        if (!line_buffer)
        {
            Py_DECREF(out_array);
            return nullptr;
        }
        free_mem = false;
    }
    else
    {
        line_buffer = PyMem_Malloc(sizeof(*line_buffer) * this->mesh.n_lines * n_cpts);
        if (!line_buffer)
        {
            Py_DECREF(out_array);
            return nullptr;
        }
        free_mem = true;
    }

    // Now I can be sure the arrays are well-behaved
    const real3_t *control_pts = PyArray_DATA(in_array);
    real3_t *out_ptr = PyArray_DATA(out_array);

    compute_line_induction(this->mesh.n_lines, this->mesh.lines, this->mesh.n_points, this->mesh.positions, n_cpts,
                           control_pts, line_buffer, tol);
    line_induction_to_surface_induction(this->mesh.n_surfaces, this->mesh.surfaces, this->mesh.n_lines, n_cpts,
                                        line_buffer, out_ptr);

    if (free_mem)
        PyMem_Free(line_buffer);

    return (PyObject *)out_array;
}

static PyObject *pydust_mesh_induction_matrix2(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Method requires 2 arguments, but was called with %u.", (unsigned)nargs);
        return nullptr;
    }
    const double tol = PyFloat_AsDouble(args[0]);
    if (PyErr_Occurred())
        return nullptr;

    PyArrayObject *const in_array =
        pydust_ensure_array(args[1], 2, (const npy_intp[2]){0, 3}, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                            NPY_FLOAT64, "Control point array");
    if (!in_array)
        return nullptr;
    const npy_intp *dims = PyArray_DIMS(in_array);
    const unsigned n_cpts = dims[0];

    PyArrayObject *out_array;

    const npy_intp out_dims[3] = {n_cpts, this->mesh.n_surfaces, 3};
    out_array = (PyArrayObject *)PyArray_SimpleNew(3, out_dims, NPY_FLOAT64);
    if (!out_array)
        return nullptr;
    Py_INCREF(out_array);

    // Now I can be sure the arrays are well-behaved
    const real3_t *control_pts = PyArray_DATA(in_array);
    real3_t *out_ptr = PyArray_DATA(out_array);

    for (unsigned s = 0; s < this->mesh.n_surfaces; ++s)
        for (unsigned c = 0; c < n_cpts; ++c)
            out_ptr[c * this->mesh.n_surfaces + s] =
                compute_mesh_surface_induction(control_pts[c], (geo_id_t){.value = s}, &this->mesh, tol);

    return (PyObject *)out_array;
}

static PyObject *pydust_line_velocities_from_point_velocities(PyObject *self, PyObject *const *args,
                                                              const Py_ssize_t nargs)
{
    // args:
    //  1.  Point velocities
    //  2.  Output array of line velocities
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Static method requires 2 arguments, but was called with %u instead.",
                     (unsigned)nargs);
        return nullptr;
    }

    const PyDust_MeshObject *primal = (PyDust_MeshObject *)self;

    PyArrayObject *const point_velocities =
        pydust_ensure_array(args[0], 2, (const npy_intp[2]){primal->mesh.n_points, 3},
                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Point velocities");
    if (!point_velocities)
        return nullptr;
    PyArrayObject *const line_buffer = pydust_ensure_array(
        args[1], 2, (const npy_intp[2]){primal->mesh.n_lines, 3},
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, NPY_FLOAT64, "Output array");
    if (!line_buffer)
        return nullptr;

    _Static_assert(3 * sizeof(npy_float64) == sizeof(real3_t));
    real3_t const *restrict velocities_in = PyArray_DATA(point_velocities);
    real3_t *restrict velocities_out = PyArray_DATA(line_buffer);

#pragma omp parallel for default(none) shared(primal, velocities_in, velocities_out)
    for (unsigned i = 0; i < primal->mesh.n_lines; ++i)
    {
        const line_t *ln = primal->mesh.lines + i;
        velocities_out[i] = real3_mul1(real3_add(velocities_in[ln->p1.value], velocities_in[ln->p2.value]), 0.5);
    }

    Py_RETURN_NONE;
}

static PyObject *pydust_line_forces(PyObject *Py_UNUSED(module), PyObject *const *args, const Py_ssize_t nargs)
{
    if (nargs != 4)
    {
        PyErr_Format(PyExc_TypeError, "Static method requires 4 arguments, but was called with %u instead.",
                     (unsigned)nargs);
        return nullptr;
    }
    if (!PyObject_TypeCheck(args[0], &pydust_mesh_type))
    {
        PyErr_Format(PyExc_TypeError, "First argument must be a mesh, but it was %R instead.", PyObject_Type(args[0]));
        return nullptr;
    }
    const PyDust_MeshObject *primal = (PyDust_MeshObject *)args[0];
    if (!PyObject_TypeCheck(args[1], &pydust_mesh_type))
    {
        PyErr_Format(PyExc_TypeError, "Second argument must be a mesh, but it was %R instead.", PyObject_Type(args[1]));
        return nullptr;
    }
    const PyDust_MeshObject *dual = (PyDust_MeshObject *)args[1];
    if (dual->mesh.n_lines != primal->mesh.n_lines || dual->mesh.n_points != primal->mesh.n_surfaces ||
        dual->mesh.n_surfaces != primal->mesh.n_points)
    {
        PyErr_Format(PyExc_ValueError,
                     "First two arguments can not be primal/dual pair, since the counts"
                     " of primal and dual objects do not match (primal is %R, dual is %R)",
                     primal, dual);
        return nullptr;
    }

    PyArrayObject *const surface_circulations =
        pydust_ensure_array(args[2], 1, (const npy_intp[1]){primal->mesh.n_surfaces},
                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Surface circulation array");
    if (!surface_circulations)
        return nullptr;

    PyArrayObject *const output_array = pydust_ensure_array(
        args[3], 2, (const npy_intp[2]){primal->mesh.n_lines, 3},
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, NPY_FLOAT64, "Output force array");
    if (!output_array)
        return nullptr;

    _Static_assert(sizeof(npy_float64) == sizeof(real_t));
    line_forces_from_surface_circulation(&primal->mesh, &dual->mesh, PyArray_DATA(output_array),
                                         PyArray_DATA(surface_circulations));

    Py_RETURN_NONE;
}

static PyObject *pydust_mesh_merge(PyObject *type, PyObject *const *args, Py_ssize_t nargs)
{
    unsigned n_surfaces = 0, n_lines = 0, n_points = 0, n_surface_entries = 0;

    for (unsigned i = 0; i < nargs; ++i)
    {
        PyObject *const o = args[i];
        if (!PyObject_TypeCheck(o, &pydust_mesh_type))
        {
            PyErr_Format(PyExc_TypeError, "Element %u in the input sequence was not a Mesh, but was instead %R", i,
                         PyObject_Type(o));
            return nullptr;
        }
        const PyDust_MeshObject *const this = (PyDust_MeshObject *)o;
        n_surfaces += this->mesh.n_surfaces;
        n_lines += this->mesh.n_lines;
        n_points += this->mesh.n_points;
        for (unsigned i_s = 0; i_s < this->mesh.n_surfaces; ++i_s)
        {
            const surface_t *const ps = this->mesh.surfaces[i_s];
            n_surface_entries += ps->n_lines;
        }
    }

    PyDust_MeshObject *const this = (PyDust_MeshObject *)((PyTypeObject *)type)->tp_alloc((PyTypeObject *)type, 0);
    if (!this)
    {
        return nullptr;
    }

    real3_t *const positions = PyObject_Malloc(sizeof *positions * n_points);
    line_t *const lines = PyObject_Malloc(sizeof *lines * n_lines);
    const surface_t **const surfaces =
        PyObject_Malloc(sizeof *surfaces * n_surfaces + (n_surfaces + n_surface_entries) * sizeof(geo_id_t));

    if (!positions || !lines || !surfaces)
    {
        PyObject_Free(surfaces);
        PyObject_Free(lines);
        PyObject_Free(positions);
        return nullptr;
    }

    unsigned cnt_pts = 0, cnt_lns = 0, cnt_surf = 0;
    real3_t *p = positions;
    line_t *l = lines;
    surface_t **s = (surface_t **)surfaces;
    geo_id_t *v = (geo_id_t *)(surfaces + n_surfaces);
    for (unsigned i = 0; i < nargs; ++i)
    {
        const PyDust_MeshObject *const m = (PyDust_MeshObject *)args[i];
        // Positions we copy
        memcpy(p, m->mesh.positions, sizeof(*p) * m->mesh.n_points);
        p += m->mesh.n_points;
        // Lines are copied, but incremented
        for (unsigned il = 0; il < m->mesh.n_lines; ++il)
        {
            const line_t *p_line = m->mesh.lines + il;
            *l = (line_t){
                .p1 = (geo_id_t){.orientation = p_line->p1.orientation, p_line->p1.value + cnt_pts},
                .p2 = (geo_id_t){.orientation = p_line->p2.orientation, p_line->p2.value + cnt_pts},
            };
            l += 1;
        }
        // Surfaces are a bit more involved
        for (unsigned is = 0; is < m->mesh.n_surfaces; ++is)
        {
            const surface_t *p_surf = m->mesh.surfaces[is];
            *(uint32_t *)v = p_surf->n_lines;
            for (unsigned il = 0; il < p_surf->n_lines; ++il)
            {
                v[1 + il] = (geo_id_t){
                    .orientation = p_surf->lines[il].orientation,
                    .value = p_surf->lines[il].value + cnt_lns,
                };
            }
            *s = (surface_t *)v;
            s += 1;
            v += (1 + p_surf->n_lines);
        }
        cnt_pts += m->mesh.n_points;
        cnt_lns += m->mesh.n_lines;
        cnt_surf += m->mesh.n_surfaces;
    }

    this->mesh = (mesh_t){
        .n_points = cnt_pts,
        .positions = positions,
        .n_lines = cnt_lns,
        .lines = lines,
        .n_surfaces = cnt_surf,
        .surfaces = surfaces,
    };

    return (PyObject *)this;
}

static PyObject *pydust_mesh_copy(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    const PyDust_MeshObject *const origin = (PyDust_MeshObject *)self;
    if (nargs > 1)
    {
        PyErr_Format(PyExc_TypeError, "Method takes either one or no arguments, but %u were given.", (unsigned)nargs);
        return nullptr;
    }

    PyArrayObject *pos_array = nullptr;
    if (nargs == 1 && !Py_IsNone(args[0]))
    {
        pos_array = (PyArrayObject *)PyArray_FromAny(args[0], PyArray_DescrFromType(NPY_FLOAT64), 2, 2,
                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, nullptr);
        if (!pos_array)
            return nullptr;
        const npy_intp *const dims = PyArray_DIMS(pos_array);
        if (dims[0] != origin->mesh.n_points || dims[1] != 3)
        {
            PyErr_Format(PyExc_ValueError,
                         "Input array did not have the same shape as the positions of the mesh"
                         " (got (%u, %u) when expecting (%u, 3)).",
                         (unsigned)dims[0], (unsigned)dims[1], origin->mesh.n_points);
            Py_DECREF(pos_array);
            return nullptr;
        }
    }

    PyDust_MeshObject *const this = (PyDust_MeshObject *)pydust_mesh_type.tp_alloc(&pydust_mesh_type, 0);
    if (!this)
    {
        Py_XDECREF(pos_array);
        return nullptr;
    }

    this->mesh.n_points = origin->mesh.n_points;
    this->mesh.n_lines = origin->mesh.n_lines;
    this->mesh.n_surfaces = origin->mesh.n_surfaces;

    unsigned lines_in_surfaces = 0;
    for (unsigned i = 0; i < origin->mesh.n_surfaces; ++i)
    {
        const surface_t *s = origin->mesh.surfaces[i];
        lines_in_surfaces += s->n_lines;
    }

    this->mesh.positions = PyObject_Malloc(sizeof(*this->mesh.positions) * this->mesh.n_points);
    this->mesh.lines = PyObject_Malloc(sizeof(*this->mesh.lines) * this->mesh.n_lines);
    this->mesh.surfaces = PyObject_Malloc(sizeof(*this->mesh.surfaces) * this->mesh.n_surfaces +
                                          sizeof(uint32_t) * (this->mesh.n_surfaces + lines_in_surfaces));
    if (!this->mesh.positions || !this->mesh.lines || !this->mesh.surfaces)
    {
        PyObject_Free(this->mesh.surfaces);
        PyObject_Free(this->mesh.lines);
        PyObject_Free(this->mesh.positions);
        Py_XDECREF(pos_array);
        return nullptr;
    }

    if (pos_array)
    {
        const npy_float64 *pos_in = PyArray_DATA(pos_array);
        _Static_assert(3 * sizeof(*pos_in) == sizeof(*this->mesh.positions));
        memcpy(this->mesh.positions, pos_in, sizeof(*this->mesh.positions) * this->mesh.n_points);
        Py_DECREF(pos_array);
        pos_array = nullptr;
    }
    else
    {
        memcpy(this->mesh.positions, origin->mesh.positions, sizeof(*this->mesh.positions) * this->mesh.n_points);
    }

    memcpy(this->mesh.lines, origin->mesh.lines, sizeof(*this->mesh.lines) * this->mesh.n_lines);

    surface_t **s = (surface_t **)this->mesh.surfaces;
    geo_id_t *v = (geo_id_t *)(this->mesh.surfaces + this->mesh.n_surfaces);

    for (unsigned is = 0; is < origin->mesh.n_surfaces; ++is)
    {
        const surface_t *p_surf = origin->mesh.surfaces[is];
        *s = (surface_t *)v;
        const unsigned n_units = (1 + p_surf->n_lines);
        memcpy(s, p_surf, sizeof(uint32_t) * n_units);
        s += 1;
        v += n_units;
    }

    return (PyObject *)this;
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
    {.ml_name = "induction_matrix",
     .ml_meth = (PyCFunction)pydust_mesh_induction_matrix,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute an induction matrix for the mesh."},
    {.ml_name = "induction_matrix2",
     .ml_meth = (PyCFunction)pydust_mesh_induction_matrix2,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute an induction matrix for the mesh, but less cool."},
    {.ml_name = "induction_matrix3",
     .ml_meth = (PyCFunction)pydust_mesh_induction_matrix3,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute an induction matrix with normals included."},
    {.ml_name = "line_velocities_from_point_velocities",
     .ml_meth = (PyCFunction)pydust_line_velocities_from_point_velocities,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute line velocities by averaging velocities at its end nodes."},
    {.ml_name = "line_velocity_to_force",
     .ml_meth = (PyCFunction)pydust_line_forces,
     .ml_flags = METH_FASTCALL | METH_STATIC,
     .ml_doc = "Compute line forces due to average velocity along it inplace."},
    {.ml_name = "merge_meshes",
     .ml_meth = (PyCFunction)pydust_mesh_merge,
     .ml_flags = METH_CLASS | METH_FASTCALL,
     .ml_doc = "Merge sequence of meshes together into a single mesh."},
    {.ml_name = "copy",
     .ml_meth = (PyCFunction)pydust_mesh_copy,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Create a copy of the mesh, with optionally changed positions."},
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
    .tp_dealloc = pydust_mesh_dealloc,
};
