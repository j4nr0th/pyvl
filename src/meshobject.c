#include "meshobject.h"
#include "core/flow_solver.h"
#include "core/mesh.h"

#include <numpy/arrayobject.h>

#include "allocator.h"

// Should be the last to be included
#include "geoidobject.h"

#include <cpyutl.h>

static PyObject *pyvl_mesh_str(PyObject *self)
{
    const PyVL_MeshObject *this = (PyVL_MeshObject *)self;
    return PyUnicode_FromFormat("Mesh(%u points, %u lines, %u surfaces)", this->mesh.n_points, this->mesh.n_lines,
                                this->mesh.n_surfaces);
}

PyDoc_STRVAR(pyvl_mesh_type_docstring, "Wrapper around cvl mesh type.");

static PyObject *pyvl_mesh_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyVL_MeshObject *this = NULL;
    unsigned n_elements = 0;
    unsigned *per_element = NULL;
    unsigned *flat_points = NULL;
    PyObject *seq = NULL;

    PyObject *root;
    unsigned n_points;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IO", (char *[3]){"n_points", "connectivity", NULL}, &n_points,
                                     &root))
    {
        return NULL;
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
            const Py_ssize_t len = PySequence_Size(PySequence_Fast_GET_ITEM(seq, i));
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
            const PyArrayObject *const idx = (PyArrayObject *)PyArray_FromAny(
                PySequence_Fast_GET_ITEM(seq, i), PyArray_DescrFromType(NPY_UINT), 1, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
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
        this = (PyVL_MeshObject *)type->tp_alloc(type, 0);
        if (!this)
        {
            goto end;
        }
        const int status = mesh_from_elements(&this->mesh, n_elements, per_element, flat_points, &CVL_OBJ_ALLOCATOR);
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

static void pyvl_mesh_dealloc(PyObject *self)
{
    PyVL_MeshObject *this = (PyVL_MeshObject *)self;
    PyObject_GC_UnTrack(self);

    CVL_OBJ_ALLOCATOR.deallocate(CVL_OBJ_ALLOCATOR.state, this->mesh.lines);
    CVL_OBJ_ALLOCATOR.deallocate(CVL_OBJ_ALLOCATOR.state, this->mesh.surface_offsets);
    CVL_OBJ_ALLOCATOR.deallocate(CVL_OBJ_ALLOCATOR.state, this->mesh.surface_lines);

    PyTypeObject *type = Py_TYPE(this);
    type->tp_free(this);
    Py_DECREF(type);
}

static PyObject *pyvl_mesh_get_n_points(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_MeshObject *this = (PyVL_MeshObject *)self;
    return PyLong_FromUnsignedLong(this->mesh.n_points);
}

static PyObject *pyvl_mesh_get_n_lines(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_MeshObject *this = (PyVL_MeshObject *)self;
    return PyLong_FromUnsignedLong(this->mesh.n_lines);
}

static PyObject *pyvl_mesh_get_n_surfaces(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_MeshObject *this = (PyVL_MeshObject *)self;
    return PyLong_FromUnsignedLong(this->mesh.n_surfaces);
}

static PyObject *pyvl_mesh_get_line_data(PyObject *self, void *Py_UNUSED(closere))
{
    const PyVL_MeshObject *const this = (PyVL_MeshObject *)self;
    _Static_assert(sizeof(*this->mesh.lines) == 2 * sizeof(npy_uint32), "Types must have the same size.");
    const npy_intp dims[2] = {this->mesh.n_lines, 2};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_UINT32, this->mesh.lines);
    if (!out)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject(out, (PyObject *)this))
    {
        Py_DECREF(out);
        return NULL;
    }
    Py_INCREF(this);

    return (PyObject *)out;
}

static PyGetSetDef pyvl_mesh_getset[] = {
    {
        .name = "n_points",
        .get = pyvl_mesh_get_n_points,
        .doc = "Number of points in the mesh",
    },
    {
        .name = "n_lines",
        .get = pyvl_mesh_get_n_lines,
        .doc = "Number of lines in the mesh",
    },
    {
        .name = "n_surfaces",
        .get = pyvl_mesh_get_n_surfaces,
        .doc = "Number of surfaces in the mesh",
    },
    {
        .name = "line_data",
        .get = pyvl_mesh_get_line_data,
        .doc = "Line connectivity of the mesh.",
    },
    {0},
};

/**
 * Helper function for bound methods and class methods of the mesh type to get the module state and type check self.
 *
 * @param defining_class Defining class passed to the method or NULL.
 * @param self Self-instance, used to type check and fallback for module state when defining class is not given.
 * @param p_this Address to receive the type-checked self.
 * @param p_state Address to receive the module state.
 * @return False on error, with an exception raise. True when successful.
 */
static bool ensure_mesh_and_state(PyTypeObject *defining_class, PyObject *self, const PyVL_MeshObject **p_this,
                                  const module_state_t **p_state)
{
    // Get the state from the defining class if possible
    const module_state_t *state = NULL;
    if (defining_class)
        state = PyType_GetModuleState(defining_class);
    else
        state = get_module_state(Py_TYPE(self));

    if (!state)
        return false;

    if (!PyObject_TypeCheck(self, state->mesh_type))
    {
        PyErr_Format(PyExc_TypeError, "self was not %s, but was %s instead.", state->mesh_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return false;
    }
    *p_this = (PyVL_MeshObject *)self;
    *p_state = state;
    return true;
}

static PyObject *pyvl_mesh_get_line_points(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                           const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_MeshObject *this;
    const module_state_t *state;
    if (!ensure_mesh_and_state(defining_class, self, &this, &state))
        return NULL;

    PyObject *o;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&o, .kwname = "i"},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    geo_id_t line_id;
    if (!pyvl_geoid_from_pyvalue(state, o, &line_id))
        return NULL;

    if (line_id.value >= this->mesh.n_lines)
    {
        PyErr_Format(PyExc_IndexError, "Index %u is our of bounds for a mesh with %u lines.", line_id.value,
                     this->mesh.n_lines);
        return NULL;
    }

    const line_t line = this->mesh.lines[line_id.value];

    PyObject *const out = PyTuple_New(2);
    if (!out)
        return NULL;

    unsigned start, end;
    if (line_id.orientation)
    {
        start = line.p2.value;
        end = line.p1.value;
    }
    else
    {
        start = line.p1.value;
        end = line.p2.value;
    }

    return cpyutl_output_create_check(CPYOUT_TYPE_TUPLE, (const cpyutl_output_t[]){
                                                             {.type = CPYOUT_TYPE_PYINT, .value_int = start},
                                                             {.type = CPYOUT_TYPE_PYINT, .value_int = end},
                                                             {0},
                                                         });
}

static PyObject *pyvl_mesh_get_surface_lines(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                             const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_MeshObject *this;
    const module_state_t *state;
    if (!ensure_mesh_and_state(defining_class, self, &this, &state))
        return NULL;

    PyObject *o;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&o, .kwname = "i"},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    geo_id_t surface_id;
    if (!pyvl_geoid_from_pyvalue(state, o, &surface_id))
        return NULL;

    if (surface_id.value >= this->mesh.n_surfaces)
    {
        PyErr_Format(PyExc_IndexError, "Index %u is our of bounds for a mesh with %u surfaces.", surface_id.value,
                     this->mesh.n_surfaces);
        return NULL;
    }

    const geo_id_t *lines;
    const unsigned n_lines = mesh_get_surface(&this->mesh, surface_id.value, &lines);
    PyObject *out = PyTuple_New(n_lines);
    if (!out)
        return NULL;

    for (unsigned i = 0; i < n_lines; ++i)
    {
        geo_id_t id = lines[surface_id.orientation ? n_lines - 1 - i : i];
        if (surface_id.orientation)
            id.orientation = !id.orientation;
        PyVL_GeoIDObject *const gid = pyvl_geoid_new(state, id);
        if (!gid)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, gid);
    }

    return out;
}

static bool ensure_mesh_and_state_noargs(PyTypeObject *defining_class, PyObject *self, PyObject *const *args,
                                         const Py_ssize_t nargs, const PyObject *kwnames,
                                         const PyVL_MeshObject **p_this, const module_state_t **p_state,
                                         const char *method_name)
{
    if (!ensure_mesh_and_state(defining_class, self, p_this, p_state))
        return false;

    if ((args != NULL && nargs != 0) || (kwnames != NULL && PyTuple_GET_SIZE(kwnames) != 0))
    {
        PyErr_Format(PyExc_TypeError, "Method \"%s\" takes no parameters.", method_name);
        return false;
    }
    return true;
}

static PyObject *pyvl_mesh_compute_dual(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                        const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_MeshObject *this;
    const module_state_t *state;
    if (!ensure_mesh_and_state_noargs(defining_class, self, args, nargs, kwnames, &this, &state, "compute_dual"))
        return NULL;

    PyVL_MeshObject *that = (PyVL_MeshObject *)state->mesh_type->tp_alloc(state->mesh_type, 0);
    if (!that)
    {
        return NULL;
    }
    const int stat = mesh_dual_from_primal(&that->mesh, &this->mesh, &CVL_OBJ_ALLOCATOR);
    if (stat != 0)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not compute dual to the mesh.");
        Py_DECREF(that);
        return NULL;
    }
    return (PyObject *)that;
}

static void cleanup_memory(PyObject *cap)
{
    void *const ptr = PyCapsule_GetPointer(cap, NULL);
    PyMem_Free(ptr);
}

static PyObject *pyvl_mesh_to_element_connectivity(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                   const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_MeshObject *this;
    const module_state_t *state;
    if (!ensure_mesh_and_state_noargs(defining_class, self, args, nargs, kwnames, &this, &state,
                                      "to_element_connectivity"))
        return NULL;

    unsigned *point_counts, *flat_points;
    const unsigned n_elements = mesh_to_elements(&this->mesh, &point_counts, &flat_points, &CVL_MEM_ALLOCATOR);
    if (n_elements != this->mesh.n_surfaces)
    {
        if (!PyErr_Occurred())
        {
            PyErr_Format(PyExc_RuntimeError, "Could not convert mesh to elements.");
        }
        return NULL;
    }
    PyObject *const cap = PyCapsule_New(point_counts, NULL, cleanup_memory);
    if (!cap)
    {
        PyMem_Free(point_counts);
        PyMem_Free(flat_points);
        return NULL;
    }
    const npy_intp n_counts = n_elements;
    PyObject *const counts_array = PyArray_SimpleNewFromData(1, &n_counts, NPY_UINT, point_counts);
    if (!counts_array)
    {
        Py_DECREF(cap);
        PyMem_Free(flat_points);
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)counts_array, cap) < 0)
    {
        Py_DECREF(counts_array);
        Py_DECREF(cap);
        PyMem_Free(flat_points);
        return NULL;
    }

    PyObject *const cap_2 = PyCapsule_New(flat_points, NULL, cleanup_memory);
    if (!cap_2)
    {
        Py_DECREF(counts_array);
        PyMem_Free(flat_points);
        return NULL;
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
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)points_array, cap_2) < 0)
    {
        Py_DECREF(counts_array);
        Py_DECREF(points_array);
        return NULL;
    }

    PyObject *out = PyTuple_Pack(2, counts_array, points_array);
    if (!out)
    {
        Py_DECREF(counts_array);
        Py_DECREF(points_array);
    }
    return out;
}

static real3_t *ensure_line_memory(const PyArrayObject *in, const unsigned n_lines, const unsigned n_cpts)
{
    if (check_input_array(in, 0, (const npy_intp[0]){}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
                          "Line computation buffer") < 0)
        return NULL;

    if (PyArray_SIZE(in) < (npy_intp)n_lines * n_cpts * 3)
    {
        PyErr_Format(PyExc_ValueError,
                     "Line computation buffer did not have space for enough elements "
                     "(required %zu, but got %zu).",
                     (size_t)n_lines * n_cpts * 3, (size_t)PyArray_SIZE(in));
        return NULL;
    }
    return PyArray_DATA(in);
}

static PyObject *pyvl_mesh_induction_matrix3(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                             const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    const PyVL_MeshObject *this;
    if (!ensure_mesh_and_state(defining_class, self, &this, &state))
        return NULL;

    PyArrayObject *pos_array, *norm_array, *in_array, *out_array = NULL, *line_buffer_opt = NULL;
    double tol;
    Py_ssize_t thrd_cnt = 1;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .kwname = "tol", .p_val = &tol},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "positions",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&pos_array},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "control_points",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&in_array},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "normals",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&norm_array},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "out",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&out_array,
                 .optional = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "line_buffer",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&line_buffer_opt,
                 .optional = true},
                {.type = CPYARG_TYPE_SSIZE, .kwname = "thread_count", .p_val = &thrd_cnt, .optional = true},
                {0}, // sentinel
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (thrd_cnt < 1)
    {
        PyErr_SetString(PyExc_ValueError, "Thread count must be at least 1.");
        return NULL;
    }

    if (check_input_array(pos_array, 2, (const npy_intp[2]){this->mesh.n_points, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Position array") < 0 ||
        check_input_array(in_array, 2, (const npy_intp[2]){0, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Control point array") < 0)
        return NULL;

    const npy_intp ndim = PyArray_NDIM(in_array);
    const npy_intp *dims = PyArray_DIMS(in_array);
    const unsigned n_cpts = dims[0];

    if (check_input_array(norm_array, ndim, dims, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                          "Normal array") < 0)
        return NULL;

    if (out_array)
    {
        if (check_input_array(out_array, 2, (const npy_intp[3]){n_cpts, this->mesh.n_surfaces}, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED, "Output array") < 0)
            return NULL;

        Py_INCREF(out_array);
    }
    else
    {
        const npy_intp out_dims[2] = {n_cpts, this->mesh.n_surfaces};
        out_array = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
        if (!out_array)
            return NULL;
    }

    bool free_mem;
    real3_t *line_buffer;
    if (line_buffer_opt)
    {
        line_buffer = ensure_line_memory(line_buffer_opt, this->mesh.n_lines, n_cpts);
        if (!line_buffer)
        {
            Py_DECREF(out_array);
            return NULL;
        }
        free_mem = false;
    }
    else
    {
        line_buffer = PyMem_Malloc(sizeof(*line_buffer) * this->mesh.n_lines * n_cpts);
        if (!line_buffer)
        {
            Py_DECREF(out_array);
            return NULL;
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
                           line_buffer, tol, thrd_cnt);
    line_induction_to_normal_surface_induction(this->mesh.n_surfaces, this->mesh.surface_offsets,
                                               this->mesh.surface_lines, this->mesh.n_lines, n_cpts, normals,
                                               line_buffer, out_ptr, thrd_cnt);
    Py_END_ALLOW_THREADS;
    if (free_mem)
        PyMem_Free(line_buffer);

    return (PyObject *)out_array;
}

static PyObject *pyvl_mesh_induction_matrix(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                            const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    const PyVL_MeshObject *this;
    if (!ensure_mesh_and_state(defining_class, self, &this, &state))
        return NULL;

    PyArrayObject *pos_array, *in_array, *out_array = NULL, *line_buffer_opt = NULL;
    double tol;
    Py_ssize_t thrd_cnt = 1;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .kwname = "tol", .p_val = &tol},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "positions",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&pos_array},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "control_points",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&in_array},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "out",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&out_array,
                 .optional = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "line_buffer",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&line_buffer_opt,
                 .optional = true},
                {.type = CPYARG_TYPE_SSIZE, .kwname = "thread_count", .p_val = &thrd_cnt, .optional = true},
                {0}, // sentinel
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (thrd_cnt < 1)
    {
        PyErr_SetString(PyExc_ValueError, "Thread count must be at least 1.");
        return NULL;
    }

    if (check_input_array(pos_array, 2, (const npy_intp[2]){this->mesh.n_points, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Position array") < 0 ||
        check_input_array(in_array, 2, (const npy_intp[2]){0, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Control point array") < 0)
        return NULL;

    const npy_intp *dims = PyArray_DIMS(in_array);
    const unsigned n_cpts = dims[0];
    const npy_intp out_dims[3] = {n_cpts, this->mesh.n_surfaces, 3};

    if (out_array)
    {
        if (check_input_array(out_array, 3, out_dims, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED, "Output array") < 0)
            return NULL;
        Py_INCREF(out_array);
    }
    else
    {
        out_array = (PyArrayObject *)PyArray_SimpleNew(3, out_dims, NPY_DOUBLE);
        if (!out_array)
            return NULL;
    }

    bool free_mem;
    real3_t *line_buffer;
    if (line_buffer_opt)
    {
        line_buffer = ensure_line_memory(line_buffer_opt, this->mesh.n_lines, n_cpts);
        if (!line_buffer)
        {
            Py_DECREF(out_array);
            return NULL;
        }
        free_mem = false;
    }
    else
    {
        line_buffer = PyMem_Malloc(sizeof(*line_buffer) * this->mesh.n_lines * n_cpts);
        if (!line_buffer)
        {
            Py_DECREF(out_array);
            return NULL;
        }
        free_mem = true;
    }

    // Now I can be sure the arrays are well-behaved
    const real3_t *control_pts = PyArray_DATA(in_array);
    const real3_t *positions = PyArray_DATA(pos_array);
    real3_t *out_ptr = PyArray_DATA(out_array);
    Py_BEGIN_ALLOW_THREADS;
    compute_line_induction(this->mesh.n_lines, this->mesh.lines, this->mesh.n_points, positions, n_cpts, control_pts,
                           line_buffer, tol, thrd_cnt);
    line_induction_to_surface_induction(this->mesh.n_surfaces, this->mesh.surface_offsets, this->mesh.surface_lines,
                                        this->mesh.n_lines, n_cpts, line_buffer, out_ptr, thrd_cnt);
    Py_END_ALLOW_THREADS;
    if (free_mem)
        PyMem_Free(line_buffer);

    return (PyObject *)out_array;
}

static PyObject *pyvl_line_velocities_from_point_velocities(PyObject *self, PyTypeObject *defining_class,
                                                            PyObject *const *args, const Py_ssize_t nargs,
                                                            const PyObject *kwnames)
{
    const PyVL_MeshObject *primal;
    const module_state_t *state;
    if (!ensure_mesh_and_state(defining_class, self, &primal, &state))
        return NULL;

    PyArrayObject *point_velocities, *line_buffer;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&point_velocities, .type_check = &PyArray_Type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&line_buffer, .type_check = &PyArray_Type},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // args:
    //  1.  Point velocities
    //  2.  Output array of line velocities
    if (nargs != 2)
    {
        PyErr_Format(PyExc_TypeError, "Static method requires 2 arguments, but was called with %u instead.",
                     (unsigned)nargs);
        return NULL;
    }

    if (check_input_array(point_velocities, 2, (const npy_intp[2]){primal->mesh.n_points, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Point velocities") < 0 ||
        check_input_array(line_buffer, 2, (const npy_intp[2]){primal->mesh.n_lines, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "Output array") < 0)
        return NULL;

    _Static_assert(3 * sizeof(npy_float64) == sizeof(real3_t), "Types must have the same size.");
    real3_t const *restrict velocities_in = PyArray_DATA(point_velocities);
    real3_t *restrict velocities_out = PyArray_DATA(line_buffer);

    unsigned i;
#pragma omp parallel for default(none) shared(primal, velocities_in, velocities_out)
    for (i = 0; i < primal->mesh.n_lines; ++i)
    {
        const line_t *ln = primal->mesh.lines + i;
        velocities_out[i] = real3_mul1(real3_add(velocities_in[ln->p1.value], velocities_in[ln->p2.value]), 0.5);
    }

    Py_RETURN_NONE;
}

static PyObject *pyvl_mesh_merge(PyTypeObject *subtype, PyObject *const *args, const Py_ssize_t nargs)
{
    const module_state_t *const state = get_module_state(subtype);
    if (!state)
        return NULL;

    unsigned n_surfaces = 0, n_lines = 0, n_surface_entries = 0;

    for (unsigned i = 0; i < (unsigned)nargs; ++i)
    {
        PyObject *const o = args[i];
        if (!PyObject_TypeCheck(o, state->mesh_type))
        {
            PyErr_Format(PyExc_TypeError, "Element %u in the input sequence was not a Mesh, but was instead %R", i,
                         Py_TYPE(o));
            return NULL;
        }
        const PyVL_MeshObject *const this = (PyVL_MeshObject *)o;
        n_surfaces += this->mesh.n_surfaces;
        n_lines += this->mesh.n_lines;
        n_surface_entries += this->mesh.surface_offsets[this->mesh.n_surfaces];
    }

    PyVL_MeshObject *const this = (PyVL_MeshObject *)subtype->tp_alloc(subtype, 0);
    if (!this)
    {
        return NULL;
    }

    line_t *const lines = PyObject_Malloc(sizeof *lines * n_lines);
    unsigned *const surface_offsets = PyObject_Malloc(sizeof *surface_offsets * (n_surfaces + 1));
    geo_id_t *const surface_lines = PyObject_Malloc(sizeof *surface_lines * n_surface_entries);

    if (!lines || !surface_offsets || !surface_lines)
    {
        PyObject_Free(surface_lines);
        PyObject_Free(surface_offsets);
        PyObject_Free(lines);
        return NULL;
    }

    unsigned cnt_pts = 0, cnt_lns = 0, cnt_surf = 0, cnt_entr = 0;
    line_t *l = lines;
    for (unsigned i = 0; i < (unsigned)nargs; ++i)
    {
        const PyVL_MeshObject *const m = (PyVL_MeshObject *)args[i];
        // Lines are copied but incremented
        for (unsigned il = 0; il < m->mesh.n_lines; ++il)
        {
            const line_t *p_line = m->mesh.lines + il;
            *l = (line_t){
                .p1 = (geo_id_t){.orientation = p_line->p1.orientation, .value = p_line->p1.value + cnt_pts},
                .p2 = (geo_id_t){.orientation = p_line->p2.orientation, .value = p_line->p2.value + cnt_pts},
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

static PyObject *pyvl_mesh_copy(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_MeshObject *origin;
    const module_state_t *state;
    if (!ensure_mesh_and_state_noargs(defining_class, self, args, nargs, kwnames, &origin, &state, "copy"))
        return NULL;

    PyVL_MeshObject *const this = (PyVL_MeshObject *)state->mesh_type->tp_alloc(state->mesh_type, 0);
    if (!this)
    {
        return NULL;
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
        return NULL;
    }

    memcpy(this->mesh.lines, origin->mesh.lines, sizeof(*origin->mesh.lines) * origin->mesh.n_lines);
    memcpy(this->mesh.surface_offsets, origin->mesh.surface_offsets,
           sizeof(*origin->mesh.surface_offsets) * (origin->mesh.n_surfaces + 1));
    memcpy(this->mesh.surface_lines, origin->mesh.surface_lines,
           sizeof(*origin->mesh.surface_lines) * (origin->mesh.surface_offsets[origin->mesh.n_surfaces]));

    return (PyObject *)this;
}

static PyObject *pyvl_mesh_line_gradient(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                         const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_MeshObject *this;
    const module_state_t *state;
    if (!ensure_mesh_and_state(defining_class, self, &this, &state))
        return NULL;
    PyArrayObject *point_values, *line_values = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&point_values, .type_check = &PyArray_Type},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&line_values,
                 .type_check = &PyArray_Type,
                 .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_input_array(point_values, 1, (const npy_intp[1]){this->mesh.n_points}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Point value array") < 0)
        return NULL;

    if (line_values)
    {
        if (check_input_array(line_values, 1, (const npy_intp[1]){this->mesh.n_lines}, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
                              "Line circulation array") < 0)
            return NULL;
        Py_INCREF(line_values);
    }
    else
    {
        const npy_intp nl = this->mesh.n_lines;
        line_values = (PyArrayObject *)PyArray_SimpleNew(1, &nl, NPY_DOUBLE);
        if (!line_values)
            return NULL;
    }

    const unsigned n_lns = this->mesh.n_lines;
    const line_t *const restrict lines = this->mesh.lines;
    const real_t *const restrict v_in = PyArray_DATA(point_values);
    real_t *const restrict v_out = PyArray_DATA(point_values);

    unsigned i;
#pragma omp parallel for default(none) shared(lines, v_in, v_out, n_lns)
    for (i = 0; i < n_lns; ++i)
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

static bool ensure_mesh_for_surface_position_calculations(PyObject *self, PyTypeObject *defining_class,
                                                          PyObject *const *args, const Py_ssize_t nargs,
                                                          const PyObject *kwnames, const PyVL_MeshObject **p_this,
                                                          PyArrayObject **p_out, const real3_t **p_pos)
{
    const PyVL_MeshObject *this;
    const module_state_t *state;
    if (!ensure_mesh_and_state(defining_class, self, &this, &state))
        return false;

    PyObject *in_any;
    PyArrayObject *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&in_any},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&out, .type_check = &PyArray_Type, .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return false;

    PyArrayObject *const in_array = (PyArrayObject *)PyArray_FromAny(in_any, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
                                                                     NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!in_array)
        return false;
    if (PyArray_DIM(in_array, 1) != 3 || (unsigned)PyArray_DIM(in_array, 0) != this->mesh.n_points)
    {
        PyErr_Format(PyExc_ValueError,
                     "Input array did not have the shape expected from the number of points in"
                     " the mesh (expected a (%u, 3) array, but got (%u, %u)).",
                     this->mesh.n_points, (unsigned)PyArray_DIM(in_array, 0), (unsigned)PyArray_DIM(in_array, 1));
        return false;
    }

    const npy_intp out_dims[2] = {this->mesh.n_surfaces, 3};
    if (out)
    {
        if (check_input_array(out, 2, out_dims, NPY_DOUBLE,
                              NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Output array") < 0)
        {
            Py_DECREF(in_any);
            return false;
        }
        Py_INCREF(out);
    }
    else
    {
        out = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
        if (!out)
        {
            Py_DECREF(in_array);
            return false;
        }
    }

    _Static_assert(sizeof(npy_float64) * 3 == sizeof(real3_t), "Binary compatibility");

    *p_this = this;
    *p_out = out;
    *p_pos = PyArray_DATA(in_array);
    return true;
}

static PyObject *pyvl_mesh_surface_normal(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                          const Py_ssize_t nargs, const PyObject *kwnames)
{
    const real3_t *positions;
    const PyVL_MeshObject *this;
    PyArrayObject *out;
    if (!ensure_mesh_for_surface_position_calculations(self, defining_class, args, nargs, kwnames, &this, &out,
                                                       &positions))
        return NULL;
    real3_t *const p_out = PyArray_DATA(out);

    for (unsigned i_surf = 0; i_surf < this->mesh.n_surfaces; ++i_surf)
    {
        p_out[i_surf] = surface_normal(positions, &this->mesh, (geo_id_t){.orientation = 0, .value = i_surf});
    }

    return (PyObject *)out;
}

static PyObject *pyvl_mesh_surface_average_vec3(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                const Py_ssize_t nargs, const PyObject *kwnames)
{

    const real3_t *positions;
    const PyVL_MeshObject *this;
    PyArrayObject *out;
    if (!ensure_mesh_for_surface_position_calculations(self, defining_class, args, nargs, kwnames, &this, &out,
                                                       &positions))
        return NULL;
    real3_t *const p_out = PyArray_DATA(out);

    for (unsigned i_surf = 0; i_surf < this->mesh.n_surfaces; ++i_surf)
    {
        p_out[i_surf] = surface_center(positions, &this->mesh, (geo_id_t){.orientation = 0, .value = i_surf});
    }

    return (PyObject *)out;
}

static PyObject *pyvl_mesh_dual_normal_criterion(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                 const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_MeshObject *this;
    const module_state_t *state;
    if (!ensure_mesh_and_state(defining_class, self, &this, &state))
        return NULL;
    double crit;
    PyArrayObject *normal_array;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &crit},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&normal_array, .type_check = &PyArray_Type},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (crit > 1.0 || crit < -1.0)
    {
        char buffer[20];
        snprintf(buffer, sizeof(buffer), "%g", crit);
        PyErr_Format(PyExc_ValueError,
                     "Dot product criterion was %s, which is not inside the allowed range of"
                     " -1.0 to +1.0.",
                     buffer);
        return NULL;
    }

    if (check_input_array(normal_array, 2, (const npy_intp[2]){this->mesh.n_points, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Normals array") < 0)
        return NULL;

    _Static_assert(sizeof(real3_t) == 3 * sizeof(npy_float64), "Types must have the same size.");

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
        return NULL;
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
            p_out[idx_out] = i_line;
            idx_out += 1;
        }
    }

    return (PyObject *)array_out;
}

static PyObject *pyvl_mesh_dual_free_edges(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                           const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    const PyVL_MeshObject *this;
    if (!ensure_mesh_and_state_noargs(defining_class, self, args, nargs, kwnames, &this, &state, "dual_free_edges"))
        return NULL;

    npy_intp n_found = 0;
    for (unsigned i_line = 0; i_line < this->mesh.n_lines; ++i_line)
    {
        const line_t *ln = this->mesh.lines + i_line;
        n_found += (ln->p1.value == INVALID_ID || ln->p2.value == INVALID_ID);
    }

    PyArrayObject *const array_out = (PyArrayObject *)PyArray_SimpleNew(1, &n_found, NPY_UINT);
    if (!array_out)
        return NULL;
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

static PyObject *pyvl_mesh_from_lines(PyObject *type, PyObject *args, PyObject *kwargs)
{
    unsigned npts;
    PyObject *arg;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IO", (char *[3]){"n_points", "connectivity", NULL}, &npts, &arg))
    {
        return NULL;
    }

    PyArrayObject *const array = (PyArrayObject *)PyArray_FromAny(arg, PyArray_DescrFromType(NPY_UINT), 2, 2,
                                                                  NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!array)
        return NULL;
    const unsigned n_lines = PyArray_DIM(array, 0);
    if (PyArray_DIM(array, 1) != 2)
    {
        PyErr_Format(PyExc_ValueError,
                     "Connectivity array must have the shape (N, 2), but instead its shape was (%u, %u).",
                     (unsigned)PyArray_DIM(array, 0), (unsigned)PyArray_DIM(array, 1));
        Py_DECREF(array);
        return NULL;
    }
    PyTypeObject *const obj_type = (PyTypeObject *)type;
    PyVL_MeshObject *const this = (PyVL_MeshObject *)obj_type->tp_alloc(obj_type, 0);
    if (!this)
    {
        Py_DECREF(array);
        return NULL;
    }

    this->mesh.n_points = npts;
    this->mesh.n_lines = n_lines;
    this->mesh.n_surfaces = 0;

    this->mesh.lines = NULL;
    this->mesh.surface_lines = NULL;
    this->mesh.surface_offsets = PyObject_Malloc(sizeof *this->mesh.surface_offsets);
    if (!this->mesh.surface_offsets)
    {
        Py_DECREF(this);
        Py_DECREF(array);
        return NULL;
    }
    this->mesh.lines = PyObject_Malloc(sizeof *this->mesh.lines * n_lines);
    if (!this->mesh.lines)
    {
        Py_DECREF(this);
        Py_DECREF(array);
        return NULL;
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
            return NULL;
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

static PyObject *pyvl_mesh_line_induction_matrix(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                 const Py_ssize_t nargs, const PyObject *kwnames)
{

    const PyVL_MeshObject *this;
    const module_state_t *state;
    if (!ensure_mesh_and_state(defining_class, self, &this, &state))
        return NULL;
    PyArrayObject *pos_array, *in_array, *out_array = NULL;
    double tol;
    Py_ssize_t thrd_cnt = 1;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .kwname = "tol", .p_val = &tol},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "positions",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&pos_array},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "control_points",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&in_array},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "out",
                 .type_check = &PyArray_Type,
                 .p_val = (void *)&out_array,
                 .optional = true},
                {.type = CPYARG_TYPE_SSIZE, .kwname = "thread_count", .p_val = &thrd_cnt, .optional = true},
                {}, // sentinel
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (thrd_cnt < 1)
    {
        PyErr_SetString(PyExc_ValueError, "Thread count must be at least 1.");
        return NULL;
    }

    if (check_input_array(pos_array, 2, (const npy_intp[2]){this->mesh.n_points, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Position array") < 0 ||
        check_input_array(in_array, 2, (const npy_intp[2]){0, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Control point array") < 0)
        return NULL;

    const npy_intp *dims = PyArray_DIMS(in_array);
    const unsigned n_cpts = dims[0];

    const npy_intp out_dims[3] = {n_cpts, this->mesh.n_lines, 3};
    if (out_array)
    {
        // If None is second arg, treat it as if it is not present at all.
        if (check_input_array(out_array, 3, out_dims, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "Output tensor") < 0)
            return NULL;
        Py_INCREF(out_array);
    }
    else
    {
        out_array = (PyArrayObject *)PyArray_SimpleNew(3, out_dims, NPY_DOUBLE);
        if (!out_array)
            return NULL;
    }

    // Now I can be sure the arrays are well-behaved
    const real3_t *control_pts = PyArray_DATA(in_array);
    const real3_t *positions = PyArray_DATA(pos_array);
    real3_t *out_ptr = PyArray_DATA(out_array);
    Py_BEGIN_ALLOW_THREADS;
    compute_line_induction(this->mesh.n_lines, this->mesh.lines, this->mesh.n_points, positions, n_cpts, control_pts,
                           out_ptr, tol, thrd_cnt);
    Py_END_ALLOW_THREADS;

    return (PyObject *)out_array;
}

static PyObject *pyvl_mesh_line_forces(PyTypeObject *subtype, PyObject *const *args, const Py_ssize_t nargs,
                                       const PyObject *kwnames)
{
    const module_state_t *const state = get_module_state(subtype);
    if (!state)
        return NULL;

    const PyVL_MeshObject *primal, *dual;
    PyArrayObject *circulation, *positions, *velocity, *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "primal",
                 .p_val = (void *)&primal,
                 .type_check = state->mesh_type},
                {.type = CPYARG_TYPE_PYTHON, .kwname = "dual", .p_val = (void *)&dual, .type_check = state->mesh_type},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "circulation",
                 .p_val = (void *)&circulation,
                 .type_check = &PyArray_Type},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "positions",
                 .p_val = (void *)&positions,
                 .type_check = &PyArray_Type},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "freestream",
                 .p_val = (void *)&velocity,
                 .type_check = &PyArray_Type},
                {.type = CPYARG_TYPE_PYTHON,
                 .kwname = "out",
                 .p_val = (void *)&out,
                 .type_check = &PyArray_Type,
                 .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    const unsigned n_lines = primal->mesh.n_lines;
    if (primal->mesh.n_points != dual->mesh.n_surfaces || n_lines != dual->mesh.n_lines ||
        primal->mesh.n_surfaces != dual->mesh.n_points)
    {
        PyErr_Format(PyExc_ValueError,
                     "Given meshes can not be dual to each other, since the number of points,"
                     "lines, and surfaces don't match as primal (%u, %u, %u) and dual (%u, %u, %u).",
                     primal->mesh.n_points, n_lines, primal->mesh.n_surfaces, dual->mesh.n_points, dual->mesh.n_lines,
                     dual->mesh.n_surfaces);
        return NULL;
    }

    if (check_input_array(circulation, 1, (const npy_intp[1]){primal->mesh.n_surfaces}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Circulation array") < 0 ||
        check_input_array(positions, 2, (const npy_intp[2]){primal->mesh.n_points, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Positions array") < 0 ||
        check_input_array(velocity, 2, (const npy_intp[2]){primal->mesh.n_points, 3}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "Velocity array") < 0)
        return NULL;

    const npy_intp out_dims[2] = {n_lines, 3};
    if (out)
    {
        if (check_input_array(out, 2, out_dims, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "Output array") < 0)
            return NULL;
        Py_INCREF(out);
    }
    else
    {
        out = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
        if (!out)
        {
            return NULL;
        }
    }
    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Types must have the same size.");
    _Static_assert(sizeof(real3_t) == 3 * sizeof(npy_float64), "Types must have the same size.");

    const real_t *const restrict cir = PyArray_DATA(circulation);
    const real3_t *const restrict pos = PyArray_DATA(positions);
    const real3_t *const restrict vel = PyArray_DATA(velocity);
    real3_t *const restrict f = PyArray_DATA(out);
    const line_t *primal_lines = primal->mesh.lines;
    const line_t *dual_lines = dual->mesh.lines;

    unsigned i_line;
#pragma omp parallel for default(none) shared(n_lines, primal_lines, dual_lines, pos, cir, vel, f)
    for (i_line = 0; i_line < n_lines; ++i_line)
    {
        const line_t primal_line = primal_lines[i_line];
        const line_t dual_line = dual_lines[i_line];

        const real3_t r_begin = pos[primal_line.p1.value];
        const real3_t r_end = pos[primal_line.p2.value];

        const real3_t dr = real3_sub(r_end, r_begin);

        real_t line_circ = 0;
        if (dual_line.p1.value != INVALID_ID)
        {
            const real_t v = cir[dual_line.p1.value];
            line_circ = dual_line.p1.orientation ? -v : +v;
        }
        if (dual_line.p2.value != INVALID_ID)
        {
            const real_t v = cir[dual_line.p2.value];
            line_circ += dual_line.p2.orientation ? -v : +v;
        }

        const real3_t avg_vel_circ =
            real3_mul1(real3_add(vel[primal_line.p1.value], vel[primal_line.p2.value]), 0.5 * line_circ);

        f[i_line] = real3_cross(dr, avg_vel_circ);
    }

    return (PyObject *)out;
}

static PyObject *pyvl_mesh_line_circulations(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                             const Py_ssize_clean_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    const PyVL_MeshObject *dual;
    if (!ensure_mesh_and_state(defining_class, self, &dual, &state))
        return NULL;

    PyArrayObject *circ_arr, *out_arr = NULL;
    Py_ssize_t n_threads = 1;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&circ_arr,
                    .kwname = "circulation",
                    .type_check = &PyArray_Type,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&out_arr,
                    .kwname = "out",
                    .type_check = &PyArray_Type,
                    .optional = true,
                },
                {
                    .type = CPYARG_TYPE_SSIZE,
                    .p_val = &n_threads,
                    .kwname = "n_threads",
                    .optional = true,
                },
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // Check thread count is valid
    if (n_threads < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Thread count cannot be negative.");
        return NULL;
    }

    // Check the circulation array is the right size
    if (check_input_array(circ_arr, 1, (const npy_intp[1]){(npy_intp)dual->mesh.n_points}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "circulation") < 0)
        return NULL;

    // Ensure we have an output array
    const npy_intp sz_out = dual->mesh.n_lines;
    if (out_arr == NULL)
    {
        // Create the output array
        out_arr = (PyArrayObject *)PyArray_EMPTY(1, &sz_out, NPY_DOUBLE, false);
        if (!out_arr)
            return NULL;
    }
    else
    {
        // Check the output array has the right size and flags
        if (check_input_array(out_arr, 1, &sz_out, NPY_DOUBLE,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, "out") < 0)
            return NULL;
        Py_INCREF(out_arr);
    }

    // Inputs validated, get the C arrays
    const real_t *const restrict circulations = PyArray_DATA(circ_arr);
    real_t *const restrict out = PyArray_DATA(out_arr);

    // For each line
#pragma omp parallel for default(none) num_threads(n_threads) shared(dual, circulations, out)
    for (unsigned i_line = 0; i_line < dual->mesh.n_lines; ++i_line)
    {
        // Find what surfaces it is based on the dual mesh
        const line_t dual_line = dual->mesh.lines[i_line];

        // Get the dual circulation
        real_t circ = 0;
        if (dual_line.p1.value != INVALID_ID)
        {
            const real_t v = circulations[dual_line.p1.value];
            circ += dual_line.p1.orientation ? -v : +v;
        }
        if (dual_line.p2.value != INVALID_ID)
        {
            const real_t v = circulations[dual_line.p2.value];
            circ += dual_line.p2.orientation ? -v : +v;
        }

        // Store the resulting circulation
        out[i_line] = circ;
    }

    return (PyObject *)out_arr;
}

static PyObject *pyvl_mesh_induction_velocity(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                              const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    const PyVL_MeshObject *this;
    if (!ensure_mesh_and_state(defining_class, self, &this, &state))
        return NULL;

    double vortex_tol;
    PyArrayObject *pos_arr, *cp_arr, *circ_arr, *out_arr = NULL;
    Py_ssize_t n_threads = 1;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {
                    .type = CPYARG_TYPE_DOUBLE,
                    .p_val = &vortex_tol,
                    .kwname = "tol",
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&pos_arr,
                    .kwname = "positions",
                    .type_check = &PyArray_Type,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&cp_arr,
                    .kwname = "control_points",
                    .type_check = &PyArray_Type,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&circ_arr,
                    .kwname = "line_circulation",
                    .type_check = &PyArray_Type,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&out_arr,
                    .kwname = "out",
                    .optional = true,
                },
                {
                    .type = CPYARG_TYPE_SSIZE,
                    .p_val = &n_threads,
                    .kwname = "n_threads",
                    .optional = true,
                },
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // Check vortex tol is valid
    if (vortex_tol < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Vortex tolerance cannot be less than zero.");
        return NULL;
    }

    // Check thread count is valid
    if (n_threads < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Thread count cannot be negative.");
        return NULL;
    }

    // Check the input array
    if (check_input_array(pos_arr, 2, (const npy_intp[2]){(npy_intp)this->mesh.n_points, 3}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "positions") < 0)
        return NULL;

    const int ndim_pos = PyArray_NDIM(pos_arr);
    if (ndim_pos < 2 || PyArray_DIM(pos_arr, ndim_pos - 1) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "Position array must have the shape (..., 3).");
        return NULL;
    }

    // Check the CP array
    if (check_input_array(cp_arr, 0, (const npy_intp[0]){}, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS,
                          "control points") < 0)
        return NULL;

    const int ndim_cp = PyArray_NDIM(cp_arr);
    if (ndim_cp < 2 || PyArray_DIM(cp_arr, ndim_cp - 1) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "Control point array must have the shape (..., 3).");
        return NULL;
    }

    // Count the control points
    unsigned cp_cnt = 1;
    for (int i = 0; i < ndim_cp - 1; ++i)
        cp_cnt *= PyArray_DIM(cp_arr, i);

    // Check the circulation array is the right size
    if (check_input_array(circ_arr, 1, (const npy_intp[1]){(npy_intp)this->mesh.n_lines}, NPY_DOUBLE,
                          NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, "circulation") < 0)
        return NULL;

    // Ensure we have an output array
    if (out_arr == NULL)
    {
        // Create the output array
        out_arr = (PyArrayObject *)PyArray_EMPTY(ndim_cp, PyArray_DIMS(cp_arr), NPY_DOUBLE, false);
        if (!out_arr)
            return NULL;
    }
    else
    {
        // Check the output array has the right size and flags
        if (check_input_array(out_arr, PyArray_NDIM(cp_arr), PyArray_DIMS(cp_arr), NPY_DOUBLE,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, "out") < 0)
            return NULL;
        Py_INCREF(out_arr);
    }

    // Inputs validated, get the C arrays
    const real3_t *const restrict positions = PyArray_DATA(pos_arr);
    const real3_t *const restrict control_points = PyArray_DATA(cp_arr);
    const real_t *const restrict circulations = PyArray_DATA(circ_arr);
    real3_t *const restrict out = PyArray_DATA(out_arr);
    // Clear the output
    memset(out, 0, sizeof(*out) * cp_cnt);

    // For each line
#pragma omp parallel for default(none) num_threads(n_threads)                                                          \
    shared(this, positions, control_points, circulations, out, vortex_tol, cp_cnt)
    for (unsigned i_line = 0; i_line < this->mesh.n_lines; ++i_line)
    {
        // Get the line circulation
        const real_t circ = circulations[i_line];
        // Rare fast path
        if (circ == 0)
            continue;

        // Get the line geometry
        const line_t line = this->mesh.lines[i_line];
        const real3_t r1 = positions[line.p1.value];
        const real3_t r2 = positions[line.p2.value];
        real3_t d = real3_sub(r2, r1);
        const real_t mag = real3_mag(d);
        // If the line is too short, skip it
        if (mag < vortex_tol)
            continue;

        // Scale d by its magnitude
        d.x /= mag;
        d.y /= mag;
        d.z /= mag;

        // For each of the target points
        for (unsigned i_cp = 0; i_cp < cp_cnt; ++i_cp)
        {
            const real3_t cp = control_points[i_cp];

            // Compute induction and scale it by circulation
            const real3_t ind = real3_mul1(compute_filament_induction(vortex_tol, r1, r2, d, cp), circ);

            // Update the result atomically
#pragma omp atomic
            out[i_cp].x += ind.x;
#pragma omp atomic
            out[i_cp].y += ind.y;
#pragma omp atomic
            out[i_cp].z += ind.z;
        }
    }

    return (PyObject *)out_arr;
}

static PyMethodDef pyvl_mesh_methods[] = {
    {
        .ml_name = "get_line_points",
        .ml_meth = (void *)pyvl_mesh_get_line_points,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "get_line_points(i: GeoID | int) -> tuple[int, int]\n"
                  "Get the indices of points that make up the line from the mesh.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "i : GeoID or int\n"
                  "    ID of the line to get the points of. If an int is given, negative value\n"
                  "    means a reverse orientation.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "int\n"
                  "    Index of the point at the start of the line.\n"
                  "\n"
                  "int\n"
                  "    Index of the point at the end of the line.\n",
    },
    {
        .ml_name = "get_surface_lines",
        .ml_meth = (void *)pyvl_mesh_get_surface_lines,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "get_surface_lines(i: GeoID | int) -> tuple[GeoID, ...]\n"
                  "Get IDs of lines that make up the surface from the mesh.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "i : GeoID or int\n"
                  "    ID of the surface to get the lines of. If an int is given, negative value\n"
                  "    means a reverse orientation.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "tuple[GeoID, ...]\n"
                  "    Tuple of IDs of lines that make up the surface. When reversed orientation is\n"
                  "    requested, the order and orientation of the lines is reversed as well.\n",
    },
    {
        .ml_name = "compute_dual",
        .ml_meth = (void *)pyvl_mesh_compute_dual,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "Create dual to the mesh.",
    },
    {
        .ml_name = "to_element_connectivity",
        .ml_meth = (void *)pyvl_mesh_to_element_connectivity,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "Convert mesh connectivity to arrays list of element lengths and indices.",
    },
    {
        .ml_name = "induction_matrix",
        .ml_meth = (void *)pyvl_mesh_induction_matrix,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
        .ml_doc = "Compute an induction matrix for the mesh.",
    },
    {
        .ml_name = "induction_matrix3",
        .ml_meth = (void *)pyvl_mesh_induction_matrix3,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
        .ml_doc = "Compute an induction matrix with normals included.",
    },
    {
        .ml_name = "line_velocities_from_point_velocities",
        .ml_meth = (void *)pyvl_line_velocities_from_point_velocities,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "Compute line velocities by averaging velocities at its end nodes.",
    },
    {
        .ml_name = "merge_meshes",
        .ml_meth = (void *)pyvl_mesh_merge,
        .ml_flags = METH_CLASS | METH_FASTCALL,
        .ml_doc = "Merge sequence of meshes together into a single mesh.",
    },
    {
        .ml_name = "copy",
        .ml_meth = (void *)pyvl_mesh_copy,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "Create a copy of the mesh.",
    },
    {
        .ml_name = "line_gradient",
        .ml_meth = (void *)pyvl_mesh_line_gradient,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "Compute line gradient from point values.",
    },
    {
        .ml_name = "surface_normal",
        .ml_meth = (void *)pyvl_mesh_surface_normal,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "Compute normals to surfaces based on point positions.",
    },
    {
        .ml_name = "surface_average_vec3",
        .ml_meth = (void *)pyvl_mesh_surface_average_vec3,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "Compute average vec3 for each surface based on point values.",
    },
    {
        .ml_name = "dual_normal_criterion",
        .ml_meth = (void *)pyvl_mesh_dual_normal_criterion,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "Find edges satisfying neighbouring normal dot product criterion.",
    },
    {
        .ml_name = "dual_free_edges",
        .ml_meth = (void *)pyvl_mesh_dual_free_edges,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "Find edges with invalid nodes (dual free edges).",
    },
    {
        .ml_name = "from_lines",
        .ml_meth = (void *)pyvl_mesh_from_lines,
        .ml_flags = METH_FASTCALL | METH_CLASS | METH_KEYWORDS,
        .ml_doc = "Create line-only mesh from line connectivity.",
    },
    {
        .ml_name = "line_induction_matrix",
        .ml_meth = (void *)pyvl_mesh_line_induction_matrix,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
        .ml_doc = "Compute an induction matrix for the mesh based on line circulations.",
    },
    {
        .ml_name = "line_forces",
        .ml_meth = (void *)pyvl_mesh_line_forces,
        .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "line_forces(\n"
                  "    primal: Mesh,\n"
                  "    dual: Mesh,\n"
                  "    circulation: in_array,\n"
                  "    positions: in_array,\n"
                  "    freestream: in_array,\n"
                  "    out: out_array | None = None,\n"
                  ") -> out_array\n"
                  "Compute forces due to reduced circulation filaments.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "primal : Mesh\n"
                  "    Primal mesh.\n"
                  "dual : Mesh\n"
                  "    Dual mesh, computed from the ``primal`` by a call to :meth:`Mesh.compute_dual()`.\n"
                  "circulation : (N,) in_array\n"
                  "    Array of surface circulations divided by :math:`2 \\pi`.\n"
                  "positions : (M, 3) in_array\n"
                  "    Positions of the primal mesh nodes.\n"
                  "freestream : (M, 3) in_array\n"
                  "    Free-stream velocity at the mesh nodes.\n"
                  "out : (K, 3) out_array, optional\n"
                  "    Optional array where to write the results to. Assumed it does not alias memory from any other\n"
                  "    arrays.\n"
                  "Returns\n"
                  "-------\n"
                  "(K, 3) out_array\n"
                  "    If ``out`` was given, it is returned as well. If not, the returned value is a newly allocated\n"
                  "    array of the correct size.\n",
    },
    {
        .ml_name = "induction_velocity",
        .ml_meth = (void *)pyvl_mesh_induction_velocity,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "induction_velocity"
                  "(tol: float, positions: numpy.typing.NDArray[numpy.double], control_points: "
                  "numpy.typing.NDArray[numpy.double], "
                  "circulation: numpy.typing.NDArray[numpy.double], out: numpy.typing.NDArray[numpy.double] | None = "
                  "None, line_buffer: "
                  "numpy.typing.NDArray[numpy.double] | None = None, thread_count: int = 1) -> "
                  "numpy.typing.NDArray[numpy.double]\n"
                  "Compute velocity induced by mesh circulation.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "tol : float\n"
                  "    Minimum distance before the induced velocity is clamped to zero.\n"
                  "\n"
                  "positions : array\n"
                  "    Positions of the geometry points. Must be an aligned, continuous (N, 3) array,\n"
                  "    where N is the number of points.\n"
                  "\n"
                  "control_points : array\n"
                  "    An (M, 3) array, which specifies the positions of M points.\n"
                  "\n"
                  "line_circulation : array\n"
                  "    Array of circulations for each of the lines.\n"
                  "\n"
                  "out : array, optional\n"
                  "    An array with enough space for M velocity vectors, one for\n"
                  "    each of the control points.\n"
                  "\n"
                  "line_buffer : array, optional\n"
                  "    An array with enough space for induction vector for each of the\n"
                  "    mesh lines.\n"
                  "\n"
                  "thread_count : int, default: 1\n"
                  "    Number of threads to use for computing the induction.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "array\n"
                  "    Resulting induction vectors in an array. If ``out`` was given, the result is\n"
                  "    written to it and another reference to it returned, otherwise a new array is\n"
                  "    created.\n",
    },
    {
        .ml_name = "line_circulations",
        .ml_meth = (void *)pyvl_mesh_line_circulations,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "line_circulations(circulation: numpy.typing.NDArray[numpy.double], out: "
                  "numpy.typing.NDArray[numpy.double] | None = None, n_threads: int = 1) -> "
                  "numpy.typing.NDArray[numpy.double]\n"
                  "Compute circulations based of lines using the dual mesh.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "circulation : array\n"
                  "    Array of surface circulation values. Must match the number of points in the\n"
                  "    dual mesh.\n"
                  "\n"
                  "out : array, optional\n"
                  "    Array used to store the output. If not given or ``None``, a new array will\n"
                  "    be created.\n"
                  "\n"
                  "n_threads : int, default: 1\n"
                  "    Number of threads to use for this calculation.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "array\n"
                  "    Line circulation values. If ``out`` was not ``None``, a reference to it is\n"
                  "    returned, otherwise a new array is returned.\n",
    },
    {0},
};

static PyObject *pyvl_mesh_rich_compare(PyObject *self, PyObject *other, const int op)
{
    const module_state_t *const state = get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    if (!PyObject_TypeCheck(other, state->mesh_type) || (op != Py_EQ && op != Py_NE))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    bool res = true;
    const PyVL_MeshObject *const this = (PyVL_MeshObject *)self;
    const PyVL_MeshObject *const that = (PyVL_MeshObject *)other;
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

    res = (op == Py_EQ ? (int)res : !res) != 0;
    if (res)
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyType_Spec pyvl_mesh_typespec = {
    .name = PYVL_CTYPE_NAME(Mesh),
    .basicsize = sizeof(PyVL_MeshObject),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_str, pyvl_mesh_str},
            {Py_tp_doc, (void *)pyvl_mesh_type_docstring},
            {Py_tp_methods, pyvl_mesh_methods},
            {Py_tp_getset, pyvl_mesh_getset},
            {Py_tp_new, pyvl_mesh_new},
            {Py_tp_dealloc, pyvl_mesh_dealloc},
            {Py_tp_richcompare, pyvl_mesh_rich_compare},
            {Py_tp_traverse, cpyutl_traverse_heap_type},
            {0},
        },
};
