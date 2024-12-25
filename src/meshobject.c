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
    unsigned n_elements = 0;
    unsigned *per_element = nullptr;
    unsigned *flat_points = nullptr;
    PyObject *seq = nullptr;

    PyObject *root;
    unsigned n_points;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IO", (char *[3]){"n_points", "connectivity", nullptr}, &n_points,
                                     &root))
    {
        return nullptr;
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
                if (v > n_points)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "Element %u had specified a point with index %u as its %u"
                                 " point, while only %u points were given.",
                                 i, v, k, n_points);
                    Py_DECREF(idx);
                    goto end;
                }
                flat_points[j + k] = v;
            }
            j += per_element[i];

            Py_DECREF(idx);
        }
        this = (PyDust_MeshObject *)type->tp_alloc(type, 0);
        if (!this)
        {
            goto end;
        }
        const int status = mesh_from_elements(&this->mesh, n_elements, per_element, flat_points, &CDUST_OBJ_ALLOCATOR);
        if (status)
        {
            PyErr_Format(PyExc_RuntimeError, "Failed creating a mesh from given indices.");
            goto end;
        }
    }
    this->mesh.n_points = n_points;

end:
    PyMem_Free(flat_points);
    PyMem_Free(per_element);
    Py_XDECREF(seq);
    return (PyObject *)this;
}

static void pydust_mesh_dealloc(PyObject *self)
{
    PyDust_MeshObject *this = (PyDust_MeshObject *)self;

    CDUST_OBJ_ALLOCATOR.deallocate(CDUST_OBJ_ALLOCATOR.state, this->mesh.lines);
    CDUST_OBJ_ALLOCATOR.deallocate(CDUST_OBJ_ALLOCATOR.state, this->mesh.surface_offsets);
    CDUST_OBJ_ALLOCATOR.deallocate(CDUST_OBJ_ALLOCATOR.state, this->mesh.surface_lines);

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
    return (PyObject *)pydust_surface_from_mesh_surface(&this->mesh, (geo_id_t){.orientation = 0, .value = i});
}

static PyObject *pydust_mesh_compute_dual(PyObject *self, PyObject *Py_UNUSED(arg))
{
    PyDust_MeshObject *that = (PyDust_MeshObject *)pydust_mesh_type.tp_alloc(&pydust_mesh_type, 0);
    if (!that)
    {
        return nullptr;
    }
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    const int stat = mesh_dual_from_primal(&that->mesh, &this->mesh, &CDUST_OBJ_ALLOCATOR);
    if (stat != 0)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not compute dual to the mesh.");
        Py_DECREF(that);
        return nullptr;
    }
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
    if (nargs != 6 && nargs != 5 && nargs != 4)
    {
        PyErr_Format(PyExc_TypeError, "Method requires 4, 5, or 6 arguments, but was called with %u.", (unsigned)nargs);
        return nullptr;
    }
    const double tol = PyFloat_AsDouble(args[0]);
    if (PyErr_Occurred())
        return nullptr;

    PyArrayObject *const pos_array =
        pydust_ensure_array(args[1], 2, (const npy_intp[2]){this->mesh.n_points, 3},
                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Position array");
    if (!pos_array)
        return nullptr;

    PyArrayObject *const in_array =
        pydust_ensure_array(args[2], 2, (const npy_intp[2]){0, 3}, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                            NPY_FLOAT64, "Control point array");
    if (!in_array)
        return nullptr;
    const npy_intp ndim = PyArray_NDIM(in_array);
    const npy_intp *dims = PyArray_DIMS(in_array);
    const unsigned n_cpts = dims[0];

    PyArrayObject *const norm_array = pydust_ensure_array(
        args[3], ndim, dims, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Normal array");
    if (!norm_array)
        return nullptr;

    PyArrayObject *out_array;
    if (nargs > 4 && !Py_IsNone(args[4]))
    {
        // If None is second arg, treat it as if it is not present at all.
        out_array = pydust_ensure_array(args[4], 2, (const npy_intp[3]){n_cpts, this->mesh.n_surfaces},
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
    if (nargs == 6 && !Py_IsNone(args[5]))
    {
        line_buffer = ensure_line_memory(args[5], this->mesh.n_lines, n_cpts);
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
    const real3_t *restrict positions = PyArray_DATA(pos_array);
    const real3_t *restrict control_pts = PyArray_DATA(in_array);
    const real3_t *restrict normals = PyArray_DATA(norm_array);
    real_t *restrict out_ptr = PyArray_DATA(out_array);

    Py_BEGIN_ALLOW_THREADS;
    compute_line_induction(this->mesh.n_lines, this->mesh.lines, this->mesh.n_points, positions, n_cpts, control_pts,
                           line_buffer, tol);
    line_induction_to_normal_surface_induction(this->mesh.n_surfaces, this->mesh.surface_offsets,
                                               this->mesh.surface_lines, this->mesh.n_lines, n_cpts, normals,
                                               line_buffer, out_ptr);
    Py_END_ALLOW_THREADS;
    if (free_mem)
        PyMem_Free(line_buffer);

    return (PyObject *)out_array;
}

static PyObject *pydust_mesh_induction_matrix(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
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

    PyArrayObject *const pos_array =
        pydust_ensure_array(args[1], 2, (const npy_intp[2]){this->mesh.n_points, 3},
                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Position");
    if (!pos_array)
        return nullptr;
    PyArrayObject *const in_array =
        pydust_ensure_array(args[2], 2, (const npy_intp[2]){0, 3}, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                            NPY_FLOAT64, "Control point array");
    if (!in_array)
        return nullptr;
    const npy_intp *dims = PyArray_DIMS(in_array);
    const unsigned n_cpts = dims[0];

    PyArrayObject *out_array;
    if (nargs > 3 && !Py_IsNone(args[3]))
    {
        // If None is second arg, treat it as if it is not present at all.
        out_array = pydust_ensure_array(args[3], 3, (const npy_intp[3]){n_cpts, this->mesh.n_surfaces, 3},
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
    const real3_t *control_pts = PyArray_DATA(in_array);
    const real3_t *positions = PyArray_DATA(pos_array);
    real3_t *out_ptr = PyArray_DATA(out_array);
    Py_BEGIN_ALLOW_THREADS;
    compute_line_induction(this->mesh.n_lines, this->mesh.lines, this->mesh.n_points, positions, n_cpts, control_pts,
                           line_buffer, tol);
    line_induction_to_surface_induction(this->mesh.n_surfaces, this->mesh.surface_offsets, this->mesh.surface_lines,
                                        this->mesh.n_lines, n_cpts, line_buffer, out_ptr);
    Py_END_ALLOW_THREADS;
    if (free_mem)
        PyMem_Free(line_buffer);

    return (PyObject *)out_array;
}

static PyObject *pydust_mesh_induction_matrix2(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
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

    PyArrayObject *const pos_array =
        pydust_ensure_array(args[1], 2, (const npy_intp[2]){this->mesh.n_points, 3},
                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Position");
    if (!pos_array)
        return nullptr;
    PyArrayObject *const in_array =
        pydust_ensure_array(args[2], 2, (const npy_intp[2]){0, 3}, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                            NPY_FLOAT64, "Control point array");
    if (!in_array)
        return nullptr;
    const npy_intp *dims = PyArray_DIMS(in_array);
    const unsigned n_cpts = dims[0];

    PyArrayObject *out_array;
    if (nargs > 3 && !Py_IsNone(args[3]))
    {
        // If None is second arg, treat it as if it is not present at all.
        out_array = pydust_ensure_array(args[3], 3, (const npy_intp[3]){n_cpts, this->mesh.n_surfaces, 3},
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
    const real3_t *restrict cpts = PyArray_DATA(in_array);
    const real3_t *restrict positions = PyArray_DATA(pos_array);
    real3_t *restrict out_ptr = PyArray_DATA(out_array);

    // compute_line_induction(this->mesh.n_lines, this->mesh.lines, this->mesh.n_points, positions, n_cpts, control_pts,
    // line_buffer, tol);

    const unsigned n_lines = this->mesh.n_lines;
    const line_t *restrict lines = this->mesh.lines;
    const unsigned n_surfaces = n_surfaces;
    const unsigned *restrict surface_offsets = this->mesh.surface_offsets;
    const geo_id_t *restrict surface_lines = this->mesh.surface_lines;
    const unsigned n_entries = this->mesh.surface_offsets[this->mesh.n_surfaces];
    const unsigned n_points = this->mesh.surface_offsets[this->mesh.n_points];
    (void)n_entries;
    (void)n_points;

    Py_BEGIN_ALLOW_THREADS;

#pragma acc data copyin(positions[0 : n_points], lines[0 : n_lines], surface_offsets[0 : n_surfaces + 1],              \
                        surface_lines[0 : n_entries]) copyout(out_ptr[0 : n_surfaces])                                 \
    create(line_buffer[0 : n_lines])
    {
#pragma acc parallel loop
        for (unsigned iln = 0; iln < n_lines; ++iln)
        {
            const line_t line = lines[iln];
            const unsigned pt1 = line.p1.value, pt2 = line.p2.value;
            const real3_t r1 = positions[pt1];
            const real3_t r2 = positions[pt2];
            real3_t direction = real3_sub(r2, r1);
            const real_t len = real3_mag(direction);
            direction.v0 /= len;
            direction.v1 /= len;
            direction.v2 /= len;
#pragma acc loop
            for (unsigned icp = 0; icp < n_cpts; ++icp)
            {
                const real3_t control_point = cpts[icp];
                if (len < tol)
                {
                    //  Filament is too short
                    line_buffer[icp * n_lines + iln] = (real3_t){};
                    continue;
                }

                const real3_t dr1 = real3_sub(control_point, r1);
                const real3_t dr2 = real3_sub(control_point, r2);

                const real_t tan_dist1 = real3_dot(direction, dr1);
                const real_t tan_dist2 = real3_dot(direction, dr2);

                const real_t norm_dist1 = real3_dot(dr1, dr1) - (tan_dist1 * tan_dist1);
                const real_t norm_dist2 = real3_dot(dr2, dr2) - (tan_dist2 * tan_dist2);

                const real_t norm_dist = sqrt((norm_dist1 + norm_dist2) / 2.0);

                if (norm_dist < tol)
                {
                    //  Filament is too short
                    line_buffer[icp * n_lines + iln] = (real3_t){};
                    continue;
                }

                const real_t vel_mag_half = (atan2(tan_dist2, norm_dist) - atan2(tan_dist1, norm_dist)) / norm_dist;
                // const real3_t dr_avg = (real3_mul1(real3_add(dr1, dr2), 0.5));
                const real3_t vel_dir = real3_mul1(real3_cross(dr1, direction), vel_mag_half);
                line_buffer[icp * n_lines + iln] = vel_dir;
            }
        }

#pragma acc parallel loop collapse(2)
        for (unsigned i_surf = 0; i_surf < n_surfaces; ++i_surf)
        {
            for (unsigned i_cp = 0; i_cp < n_cpts; ++i_cp)
            {
                real3_t res = {};
                for (unsigned i_ln = surface_offsets[i_surf]; i_ln < surface_offsets[i_surf + 1]; ++i_ln)
                {
                    const geo_id_t ln_id = surface_lines[i_ln];
                    if (ln_id.orientation)
                    {
                        res = real3_sub(res, line_buffer[i_cp * n_lines + ln_id.value]);
                    }
                    else
                    {
                        res = real3_add(res, line_buffer[i_cp * n_lines + ln_id.value]);
                    }
                }
                // printf("Surface %u at CP %u has induction (%g, %g, %g)\n", i_surf, i_cp, res.x, res.y, res.z);
                out_ptr[i_cp * n_surfaces + i_surf] = res;
            }
        }
    }

    Py_END_ALLOW_THREADS;

    if (free_mem)
        PyMem_Free(line_buffer);

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

static PyObject *pydust_mesh_merge(PyObject *type, PyObject *const *args, Py_ssize_t nargs)
{
    unsigned n_surfaces = 0, n_lines = 0, n_points = 0, n_surface_entries = 0;

    for (unsigned i = 0; i < nargs; ++i)
    {
        PyObject *const o = args[i];
        if (!PyObject_TypeCheck(o, &pydust_mesh_type))
        {
            PyErr_Format(PyExc_TypeError, "Element %u in the input sequence was not a Mesh, but was instead %R", i,
                         Py_TYPE(o));
            return nullptr;
        }
        const PyDust_MeshObject *const this = (PyDust_MeshObject *)o;
        n_surfaces += this->mesh.n_surfaces;
        n_lines += this->mesh.n_lines;
        n_points += this->mesh.n_points;
        n_surface_entries += this->mesh.surface_offsets[this->mesh.n_surfaces];
    }

    PyDust_MeshObject *const this = (PyDust_MeshObject *)((PyTypeObject *)type)->tp_alloc((PyTypeObject *)type, 0);
    if (!this)
    {
        return nullptr;
    }

    line_t *const lines = PyObject_Malloc(sizeof *lines * n_lines);
    unsigned *const surface_offsets = PyObject_Malloc(sizeof *surface_offsets * (n_surfaces + 1));
    geo_id_t *const surface_lines = PyObject_Malloc(sizeof *surface_lines * n_surface_entries);

    if (!lines || !surface_offsets || !surface_lines)
    {
        PyObject_Free(surface_lines);
        PyObject_Free(surface_offsets);
        PyObject_Free(lines);
        return nullptr;
    }

    unsigned cnt_pts = 0, cnt_lns = 0, cnt_surf = 0, cnt_entr = 0;
    line_t *l = lines;
    for (unsigned i = 0; i < nargs; ++i)
    {
        const PyDust_MeshObject *const m = (PyDust_MeshObject *)args[i];
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
        // Surfaces are also copied with increments
        for (unsigned is = 0; is < m->mesh.n_surfaces; ++is)
        {
            surface_offsets[cnt_surf + is] = m->mesh.surface_offsets[is] + cnt_entr;
        }
        for (unsigned is = 0; is < m->mesh.surface_offsets[m->mesh.n_surfaces]; ++is)
        {
            const geo_id_t original_line = m->mesh.surface_lines[is];
            surface_lines[cnt_entr + is] =
                (geo_id_t){.orientation = original_line.orientation, .value = original_line.value + cnt_lns};
        }
        cnt_pts += m->mesh.n_points;
        cnt_lns += m->mesh.n_lines;
        cnt_surf += m->mesh.n_surfaces;
        cnt_entr += m->mesh.surface_offsets[m->mesh.n_surfaces];
    }
    surface_offsets[n_surfaces] = cnt_entr;
    this->mesh = (mesh_t){
        .n_points = cnt_pts,
        .n_lines = cnt_lns,
        .lines = lines,
        .n_surfaces = cnt_surf,
        .surface_offsets = surface_offsets,
        .surface_lines = surface_lines,
    };

    return (PyObject *)this;
}

static PyObject *pydust_mesh_copy(PyObject *self, PyObject *Py_UNUSED(args))
{
    const PyDust_MeshObject *const origin = (PyDust_MeshObject *)self;

    PyDust_MeshObject *const this = (PyDust_MeshObject *)pydust_mesh_type.tp_alloc(&pydust_mesh_type, 0);
    if (!this)
    {
        return nullptr;
    }

    this->mesh.n_points = origin->mesh.n_points;
    this->mesh.n_lines = origin->mesh.n_lines;
    this->mesh.n_surfaces = origin->mesh.n_surfaces;

    this->mesh.lines = PyObject_Malloc(sizeof(*origin->mesh.lines) * origin->mesh.n_lines);
    this->mesh.surface_offsets = PyObject_Malloc(sizeof(*origin->mesh.surface_offsets) * (origin->mesh.n_surfaces + 1));
    this->mesh.surface_lines =
        PyObject_Malloc(sizeof(*origin->mesh.surface_lines) * (origin->mesh.surface_offsets[origin->mesh.n_surfaces]));
    if (!this->mesh.surface_offsets || !this->mesh.lines || !this->mesh.surface_lines)
    {
        PyObject_Free(this->mesh.surface_lines);
        PyObject_Free(this->mesh.lines);
        PyObject_Free(this->mesh.surface_lines);
        return nullptr;
    }

    memcpy(this->mesh.lines, origin->mesh.lines, sizeof(*origin->mesh.lines) * origin->mesh.n_lines);
    memcpy(this->mesh.surface_offsets, origin->mesh.surface_offsets,
           sizeof(*origin->mesh.surface_offsets) * (origin->mesh.n_surfaces + 1));
    memcpy(this->mesh.surface_lines, origin->mesh.surface_lines,
           sizeof(*origin->mesh.surface_lines) * (origin->mesh.surface_offsets[origin->mesh.n_surfaces]));

    return (PyObject *)this;
}

static PyObject *pydust_mesh_line_gradient(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    // Arguments:
    // self - mesh
    // 0 - point value array
    // 1 - output line array (optional)
    if (nargs < 1 || nargs > 2)
    {
        PyErr_Format(PyExc_TypeError, "Function takes 1 to 2 arguments, but %u were given.", (unsigned)nargs);
        return nullptr;
    }
    const PyDust_MeshObject *const this = (PyDust_MeshObject *)self;
    PyArrayObject *const point_values =
        pydust_ensure_array(args[1], 1, (const npy_intp[1]){this->mesh.n_points},
                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Point value array");
    if (!point_values)
        return nullptr;

    PyArrayObject *line_values = nullptr;
    if (nargs == 2 || Py_IsNone(args[1]))
    {
        line_values = pydust_ensure_array(args[2], 1, (const npy_intp[1]){this->mesh.n_lines},
                                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, NPY_FLOAT64,
                                          "Line circulation array");
        Py_XINCREF(line_values);
    }
    else
    {
        const npy_intp nl = this->mesh.n_lines;
        line_values = (PyArrayObject *)PyArray_SimpleNew(1, &nl, NPY_FLOAT64);
    }
    if (!line_values)
        return nullptr;

    const unsigned n_lns = this->mesh.n_lines;
    const line_t *const restrict lines = this->mesh.lines;
    const real_t *const restrict v_in = PyArray_DATA(point_values);
    real_t *const restrict v_out = PyArray_DATA(point_values);

#pragma omp parallel for default(none) shared(lines, v_in, v_out, n_lns)
    for (unsigned i = 0; i < n_lns; ++i)
    {
        real_t x = 0;
        const line_t ln = lines[i];
        if (ln.p1.value != INVALID_ID)
        {
            x -= v_in[ln.p1.value];
        }
        if (ln.p2.value != INVALID_ID)
        {
            x += v_in[ln.p2.value];
        }
        v_out[i] = x;
    }

    return (PyObject *)line_values;
}

static PyObject *pydust_mesh_surface_normal(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 1 && nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Function must be called with either 1 or 2 arguments, but %u were given.",
                     (unsigned)nargs);
        return nullptr;
    }
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;

    PyArrayObject *const in_array = (PyArrayObject *)PyArray_FromAny(
        args[0], PyArray_DescrFromType(NPY_FLOAT64), 2, 2, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, nullptr);
    if (!in_array)
        return nullptr;
    if (PyArray_DIM(in_array, 1) != 3 || PyArray_DIM(in_array, 0) != this->mesh.n_points)
    {
        PyErr_Format(PyExc_ValueError,
                     "Input array did not have the shape expected from the number of points in"
                     " the mesh (expected a (%u, 3) array, but got (%u, %u)).",
                     this->mesh.n_points, (unsigned)PyArray_DIM(in_array, 0), (unsigned)PyArray_DIM(in_array, 1));
        return nullptr;
    }

    const npy_intp out_dims[2] = {this->mesh.n_surfaces, 3};
    PyArrayObject *out;
    if (nargs == 2 && !Py_IsNone(args[1]))
    {
        out =
            pydust_ensure_array(args[1], 2, out_dims, NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                                NPY_FLOAT64, "Output array");
        Py_INCREF(out);
    }
    else
    {
        out = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_FLOAT64);
    }
    if (!out)
    {
        Py_DECREF(in_array);
        return nullptr;
    }

    _Static_assert(sizeof(npy_float64) * 3 == sizeof(real3_t), "Binary compatibility");

    real3_t *const p_out = PyArray_DATA(out);
    real3_t *const positions = PyArray_DATA(in_array);

    for (unsigned i_surf = 0; i_surf < this->mesh.n_surfaces; ++i_surf)
    {
        p_out[i_surf] = surface_normal(positions, &this->mesh, (geo_id_t){.orientation = 0, .value = i_surf});
    }

    return (PyObject *)out;
}

static PyObject *pydust_mesh_surface_average_vec3(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 1 && nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Function must be called with either 1 or 2 arguments, but %u were given.",
                     (unsigned)nargs);
        return nullptr;
    }
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;

    PyArrayObject *const in_array = (PyArrayObject *)PyArray_FromAny(
        args[0], PyArray_DescrFromType(NPY_FLOAT64), 2, 2, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, nullptr);
    if (!in_array)
        return nullptr;
    if (PyArray_DIM(in_array, 1) != 3 || PyArray_DIM(in_array, 0) != this->mesh.n_points)
    {
        PyErr_Format(PyExc_ValueError,
                     "Input array did not have the shape expected from the number of points in"
                     " the mesh (expected a (%u, 3) array, but got (%u, %u)).",
                     this->mesh.n_points, (unsigned)PyArray_DIM(in_array, 0), (unsigned)PyArray_DIM(in_array, 1));
        return nullptr;
    }

    const npy_intp out_dims[2] = {this->mesh.n_surfaces, 3};
    PyArrayObject *out;
    if (nargs == 2 && !Py_IsNone(args[1]))
    {
        out =
            pydust_ensure_array(args[1], 2, out_dims, NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                                NPY_FLOAT64, "Output array");
        Py_INCREF(out);
    }
    else
    {
        out = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_FLOAT64);
    }
    if (!out)
    {
        Py_DECREF(in_array);
        return nullptr;
    }

    _Static_assert(sizeof(npy_float64) * 3 == sizeof(real3_t), "Binary compatibility");

    real3_t *const p_out = PyArray_DATA(out);
    real3_t *const positions = PyArray_DATA(in_array);

    for (unsigned i_surf = 0; i_surf < this->mesh.n_surfaces; ++i_surf)
    {
        p_out[i_surf] = surface_center(positions, &this->mesh, (geo_id_t){.orientation = 0, .value = i_surf});
    }

    return (PyObject *)out;
}

static PyObject *pydust_mesh_dual_normal_criterion(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Function takes 2 arguments, but %u were given.", (unsigned)nargs);
        return nullptr;
    }

    const real_t crit = PyFloat_AsDouble(args[0]);
    if (PyErr_Occurred())
        return nullptr;

    if (crit > 1.0 || crit < -1.0)
    {
        char buffer[20];
        snprintf(buffer, sizeof(buffer), "%g", crit);
        PyErr_Format(PyExc_ValueError,
                     "Dot product criterion was %s, which is not inside the allowed range of"
                     " -1.0 to +1.0.",
                     buffer);
        return nullptr;
    }

    const PyDust_MeshObject *const this = (PyDust_MeshObject *)self;

    PyArrayObject *const normal_array =
        pydust_ensure_array(args[1], 2, (const npy_intp[2]){this->mesh.n_points, 3},
                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Normals array");
    _Static_assert(sizeof(real3_t) == 3 * sizeof(npy_float64));

    if (!normal_array)
        return nullptr;

    const real3_t *restrict normals = PyArray_DATA(normal_array);

    npy_intp n_found = 0;
    for (unsigned i_line = 0; i_line < this->mesh.n_lines; ++i_line)
    {
        const line_t *ln = this->mesh.lines + i_line;
        if (ln->p1.value == INVALID_ID || ln->p2.value == INVALID_ID)
            continue;
        const real_t dp = real3_dot(normals[ln->p1.value], normals[ln->p2.value]);
        n_found += (dp < crit);
    }

    PyArrayObject *const array_out = (PyArrayObject *)PyArray_SimpleNew(1, &n_found, NPY_UINT);
    if (!array_out)
        return nullptr;
    npy_intp idx_out = 0;
    npy_uint *restrict p_out = PyArray_DATA(array_out);
    for (unsigned i_line = 0; i_line < this->mesh.n_lines && idx_out < n_found; ++i_line)
    {
        const line_t *ln = this->mesh.lines + i_line;
        if (ln->p1.value == INVALID_ID || ln->p2.value == INVALID_ID)
            continue;
        const real_t dp = real3_dot(normals[ln->p1.value], normals[ln->p2.value]);
        if (dp < crit)
        {
            p_out[idx_out] = (npy_uint)i_line;
            idx_out += 1;
        }
    }

    return (PyObject *)array_out;
}

static PyObject *pydust_mesh_dual_free_edges(PyObject *self, PyObject *Py_UNUSED(args))
{
    const PyDust_MeshObject *const this = (PyDust_MeshObject *)self;

    npy_intp n_found = 0;
    for (unsigned i_line = 0; i_line < this->mesh.n_lines; ++i_line)
    {
        const line_t *ln = this->mesh.lines + i_line;
        n_found += (ln->p1.value == INVALID_ID || ln->p2.value == INVALID_ID);
    }

    PyArrayObject *const array_out = (PyArrayObject *)PyArray_SimpleNew(1, &n_found, NPY_UINT);
    if (!array_out)
        return nullptr;
    npy_intp idx_out = 0;
    npy_uint *restrict p_out = PyArray_DATA(array_out);
    for (unsigned i_line = 0; i_line < this->mesh.n_lines && idx_out < n_found; ++i_line)
    {
        const line_t *ln = this->mesh.lines + i_line;
        if (ln->p1.value == INVALID_ID || ln->p2.value == INVALID_ID)
        {
            p_out[idx_out] = (npy_uint)i_line;
            idx_out += 1;
        }
    }

    return (PyObject *)array_out;
}

static PyObject *pydust_mesh_from_lines(PyObject *type, PyObject *args, PyObject *kwargs)
{
    unsigned npts;
    PyObject *arg;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IO", (char *[3]){"n_points", "connectivity", nullptr}, &npts, &arg))
    {
        return nullptr;
    }

    PyArrayObject *const array = (PyArrayObject *)PyArray_FromAny(arg, PyArray_DescrFromType(NPY_UINT), 2, 2,
                                                                  NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, nullptr);
    if (!array)
        return nullptr;
    const unsigned n_lines = PyArray_DIM(array, 0);
    if (PyArray_DIM(array, 1) != 2)
    {
        PyErr_Format(PyExc_ValueError,
                     "Connectivity array must have the shape (N, 2), but instead its shape was (%u, %u).",
                     (unsigned)PyArray_DIM(array, 0), (unsigned)PyArray_DIM(array, 1));
        Py_DECREF(array);
        return nullptr;
    }
    PyTypeObject *const obj_type = (PyTypeObject *)type;
    PyDust_MeshObject *const this = (PyDust_MeshObject *)obj_type->tp_alloc(obj_type, 0);
    if (!this)
    {
        Py_DECREF(array);
        return nullptr;
    }

    this->mesh.n_points = npts;
    this->mesh.n_lines = n_lines;
    this->mesh.n_surfaces = 0;

    this->mesh.lines = nullptr;
    this->mesh.surface_lines = nullptr;
    this->mesh.surface_offsets = PyObject_Malloc(sizeof *this->mesh.surface_offsets);
    if (!this->mesh.surface_offsets)
    {
        Py_DECREF(this);
        Py_DECREF(array);
        return nullptr;
    }
    this->mesh.lines = PyObject_Malloc(sizeof *this->mesh.lines * n_lines);
    if (!this->mesh.lines)
    {
        Py_DECREF(this);
        Py_DECREF(array);
        return nullptr;
    }

    const npy_uint32 *restrict p_in = PyArray_DATA(array);
    for (unsigned i_ln = 0; i_ln < n_lines; ++i_ln)
    {
        if (p_in[0] >= npts || p_in[1] >= npts)
        {
            PyErr_Format(PyExc_ValueError, "Line %u has points (%u, %u), but there were only %u points specified.",
                         (unsigned)p_in[0], (unsigned)p_in[1], npts);
            Py_DECREF(this);
            Py_DECREF(array);
            return nullptr;
        }
        this->mesh.lines[i_ln] = (line_t){
            .p1 = {.orientation = 0, .value = p_in[0]},
            .p2 = {.orientation = 0, .value = p_in[1]},
        };
        p_in += 2;
    }
    Py_DECREF(array);

    return (PyObject *)this;
}

static PyObject *pydust_mesh_line_induction_matrix(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    const PyDust_MeshObject *this = (PyDust_MeshObject *)self;
    if (nargs != 4 && nargs != 3)
    {
        PyErr_Format(PyExc_TypeError, "Method requires 3, or 4 arguments, but was called with %u.", (unsigned)nargs);
        return nullptr;
    }
    const double tol = PyFloat_AsDouble(args[0]);
    if (PyErr_Occurred())
        return nullptr;

    PyArrayObject *const pos_array =
        pydust_ensure_array(args[1], 2, (const npy_intp[2]){this->mesh.n_points, 3},
                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NPY_FLOAT64, "Position array");
    if (!pos_array)
        return nullptr;
    PyArrayObject *const in_array =
        pydust_ensure_array(args[2], 2, (const npy_intp[2]){0, 3}, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                            NPY_FLOAT64, "Control point array");
    if (!in_array)
        return nullptr;
    const npy_intp *dims = PyArray_DIMS(in_array);
    const unsigned n_cpts = dims[0];

    PyArrayObject *out_array;
    if (nargs > 3 && !Py_IsNone(args[3]))
    {
        // If None is second arg, treat it as if it is not present at all.
        out_array = pydust_ensure_array(args[3], 3, (const npy_intp[3]){n_cpts, this->mesh.n_lines, 3},
                                        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, NPY_FLOAT64,
                                        "Output tensor");
        if (!out_array)
            return nullptr;
        Py_INCREF(out_array);
    }
    else
    {
        const npy_intp out_dims[3] = {n_cpts, this->mesh.n_lines, 3};
        out_array = (PyArrayObject *)PyArray_SimpleNew(3, out_dims, NPY_FLOAT64);
        if (!out_array)
            return nullptr;
    }

    // Now I can be sure the arrays are well-behaved
    const real3_t *control_pts = PyArray_DATA(in_array);
    const real3_t *positions = PyArray_DATA(pos_array);
    real3_t *out_ptr = PyArray_DATA(out_array);
    Py_BEGIN_ALLOW_THREADS;
    compute_line_induction(this->mesh.n_lines, this->mesh.lines, this->mesh.n_points, positions, n_cpts, control_pts,
                           out_ptr, tol);
    Py_END_ALLOW_THREADS;

    return (PyObject *)out_array;
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
     .ml_meth = (void *)pydust_mesh_induction_matrix,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute an induction matrix for the mesh."},
    {.ml_name = "induction_matrix2",
     .ml_meth = (void *)pydust_mesh_induction_matrix2,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute an induction matrix for the mesh."},
    {.ml_name = "induction_matrix3",
     .ml_meth = (void *)pydust_mesh_induction_matrix3,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute an induction matrix with normals included."},
    {.ml_name = "line_velocities_from_point_velocities",
     .ml_meth = (void *)pydust_line_velocities_from_point_velocities,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute line velocities by averaging velocities at its end nodes."},
    {.ml_name = "merge_meshes",
     .ml_meth = (void *)pydust_mesh_merge,
     .ml_flags = METH_CLASS | METH_FASTCALL,
     .ml_doc = "Merge sequence of meshes together into a single mesh."},
    {.ml_name = "copy", .ml_meth = pydust_mesh_copy, .ml_flags = METH_NOARGS, .ml_doc = "Create a copy of the mesh."},
    {.ml_name = "line_gradient",
     .ml_meth = (void *)pydust_mesh_line_gradient,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute line gradient from point values."},
    {.ml_name = "surface_normal",
     .ml_meth = (void *)pydust_mesh_surface_normal,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute normals to surfaces based on point positions."},
    {.ml_name = "surface_average_vec3",
     .ml_meth = (void *)pydust_mesh_surface_average_vec3,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute average vec3 for each surface based on point values."},
    {.ml_name = "dual_normal_criterion",
     .ml_meth = (void *)pydust_mesh_dual_normal_criterion,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Find edges satisfying neighbouring normal dot product criterion."},
    {.ml_name = "dual_free_edges",
     .ml_meth = pydust_mesh_dual_free_edges,
     .ml_flags = METH_NOARGS,
     .ml_doc = "Find edges with invalid nodes (dual free edges)."},
    {.ml_name = "from_lines",
     .ml_meth = (void *)pydust_mesh_from_lines,
     .ml_flags = METH_VARARGS | METH_CLASS | METH_KEYWORDS,
     .ml_doc = "Create line-only mesh from line connectivity."},
    {.ml_name = "line_induction_matrix",
     .ml_meth = (void *)pydust_mesh_line_induction_matrix,
     .ml_flags = METH_FASTCALL,
     .ml_doc = "Compute an induction matrix for the mesh based on line circulations."},
    {},
};

static PyObject *pydust_mesh_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (!PyObject_TypeCheck(other, &pydust_mesh_type) || (op != Py_EQ && op != Py_NE))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    bool res = true;
    const PyDust_MeshObject *const this = (PyDust_MeshObject *)self;
    const PyDust_MeshObject *const that = (PyDust_MeshObject *)other;
    if (this->mesh.n_points != that->mesh.n_points || this->mesh.n_lines != that->mesh.n_lines ||
        this->mesh.n_surfaces != that->mesh.n_surfaces ||
        memcmp(this->mesh.lines, that->mesh.lines, sizeof(*this->mesh.lines) * this->mesh.n_lines) != 0 ||
        memcmp(this->mesh.surface_offsets, that->mesh.surface_offsets,
               sizeof(*this->mesh.surface_offsets) * (this->mesh.n_surfaces + 1)) != 0 ||
        memcmp(this->mesh.surface_lines, that->mesh.surface_lines,
               sizeof(*this->mesh.surface_lines) * this->mesh.surface_offsets[this->mesh.n_surfaces]) != 0)
    {
        res = false;
    }

    res = (op == Py_EQ) ? res : !res;
    if (res)
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

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
    .tp_richcompare = pydust_mesh_rich_compare,
};
