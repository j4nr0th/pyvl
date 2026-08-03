/*
 * Python wrapper for the OpenCL tree backend.
 *
 * Exposes:
 *   create_backend(device="gpu"|"cpu", precision="fp64"|"fp32", ...)
 *       -> CLBackend
 *   CLBackend.build_tree(coords, values, ...) -> CLTreeBuild   (future)
 *   CLBackend.rebuild(tree, coords, values)   -> CLTreeBuild
 *   CLTree.eval(points, mode="tree_code"|"direct", theta=...) -> CLTreeEval (future)
 *   CLTree.eval_sources(...)                  -> CLTreeEval
 *   CLTreeBuild.result() / .wait() / .done()  (blocks until the build finishes)
 *   CLTreeEval.result() / .wait() / .done()   (blocks until the eval finishes)
 *
 * The futures follow the "enqueue now, block at .result()" pattern: the
 * C-side begin/finish split is mapped so that begin enqueues the async
 * device work and result() runs the blocking host-side steps.
 */

#include "opencltreeobject.h"
#include "allocator.h"

#include <numpy/arrayobject.h>
// this goes last
#include <cpyutl.h>

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Convert a (..., 3) array-like to a contiguous double array and
 *        return the number of "points" (product of leading dims).
 *
 * The caller owns the returned PyArrayObject reference. On failure NULL is
 * returned and a Python exception is set.
 */
static PyArrayObject *cl_flatten_points(PyObject *obj, const char *arg_name, size_t *p_n)
{
    PyArrayObject *arr =
        (PyArrayObject *)PyArray_FROMANY(obj, NPY_DOUBLE, 2, INT_MAX, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!arr)
        return NULL;

    const int ndim = PyArray_NDIM(arr);
    if (PyArray_DIM(arr, ndim - 1) != 3)
    {
        PyErr_Format(PyExc_ValueError, "%s: last axis must have size 3, got %" NPY_INTP_FMT ".", arg_name,
                     PyArray_DIM(arr, ndim - 1));
        Py_DECREF(arr);
        return NULL;
    }

    size_t n = 1;
    for (int i = 0; i < ndim - 1; ++i)
        n *= PyArray_DIM(arr, i);
    *p_n = n;
    return arr;
}

/**
 * @brief Resolve the "device" selector to a discovered OpenCL device.
 *
 * @param device_name "gpu", "cpu", or "any".
 * @param out_device  Discovered device (or NULL on error).
 * @return true on success (out_device valid), false with a Python exception set.
 */
static bool cl_resolve_device(const char *device_name, cvl_cl_device_t *out_device)
{
    cvl_cl_status_t status;
    if (strcmp(device_name, "gpu") == 0)
    {
        status = cvl_cl_device_first_gpu(out_device);
    }
    else if (strcmp(device_name, "cpu") == 0)
    {
        status = cvl_cl_device_first_cpu(out_device);
    }
    else
    {
        status = cvl_cl_device_first_gpu(out_device);
        if (status != CVL_CL_SUCCESS)
            status = cvl_cl_device_first_cpu(out_device);
    }

    if (status != CVL_CL_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "No OpenCL %s device found: %s", device_name, cvl_cl_status_str(status));
        return false;
    }
    return true;
}

/**
 * @brief Map a cvl_cl_status_t to a Python exception and return NULL.
 *
 * @param status C status code.
 * @param what   Human-readable operation name for the message.
 * @return NULL (always).
 */
static PyObject *cl_raise_status(cvl_cl_status_t status, const char *what)
{
    switch (status)
    {
    case CVL_CL_ERR_INVALID_PARAM:
        PyErr_Format(PyExc_ValueError, "%s: invalid parameter (%s).", what, cvl_cl_status_str(status));
        break;
    case CVL_CL_ERR_BUFFER_SIZE:
        PyErr_Format(PyExc_ValueError, "%s: buffer too small (%s).", what, cvl_cl_status_str(status));
        break;
    case CVL_CL_ERR_MEMORY:
        PyErr_Format(PyExc_MemoryError, "%s: out of memory (%s).", what, cvl_cl_status_str(status));
        break;
    case CVL_CL_ERR_NOT_FOUND:
        PyErr_Format(PyExc_RuntimeError, "%s: required kernel or device not found (%s).", what,
                     cvl_cl_status_str(status));
        break;
    default:
        PyErr_Format(PyExc_RuntimeError, "%s failed: %s.", what, cvl_cl_status_str(status));
        break;
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  CLBackend                                                          */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(pyvl_cl_backend_type_docstring,
             "CLBackend(device='gpu', precision='fp64', order=4, critical_particle_count=4, max_depth=20)\n"
             "OpenCL compute backend that owns device resources and creates trees.\n"
             "\n"
             "The backend compiles the OpenCL kernels once and owns the device context,\n"
             "queue, and all buffers.  Trees created by ``build_tree()`` reuse the\n"
             "backend's device buffers across rebuilds (grow-only).\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "device : str, default 'gpu'\n"
             "    OpenCL device type: ``'gpu'``, ``'cpu'``, or ``'any'``.\n"
             "precision : str, default 'fp64'\n"
             "    Computation precision: ``'fp64'`` or ``'fp32'``.\n"
             "order : int, default 4\n"
             "    Multipole expansion order.\n"
             "critical_particle_count : int, default 4\n"
             "    Subdivision threshold.\n"
             "max_depth : int, default 20\n"
             "    Maximum octree depth (≤ 21).\n"
             "\n"
             "See Also\n"
             "--------\n"
             "create_backend : Module-level helper to construct a backend.\n"
             "CLTree : Tree produced by ``build_tree()``.\n");

static PyObject *pyvl_cl_backend_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyVL_CLBackendObject *self = (PyVL_CLBackendObject *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    const char *device_name = "gpu";
    const char *precision_name = "fp64";
    Py_ssize_t order = 4, critical = 4, max_depth = 20;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|ssnnn",
            (char *[6]){"device", "precision", "order", "critical_particle_count", "max_depth", NULL}, &device_name,
            &precision_name, &order, &critical, &max_depth))
    {
        Py_DECREF(self);
        return NULL;
    }

    /* Validate. */
    if (strcmp(precision_name, "fp64") != 0 && strcmp(precision_name, "fp32") != 0)
    {
        PyErr_Format(PyExc_ValueError, "precision must be 'fp64' or 'fp32', got '%s'.", precision_name);
        Py_DECREF(self);
        return NULL;
    }
    if (order < 1)
    {
        PyErr_SetString(PyExc_ValueError, "order must be >= 1.");
        Py_DECREF(self);
        return NULL;
    }
    if (critical < 1)
    {
        PyErr_SetString(PyExc_ValueError, "critical_particle_count must be >= 1.");
        Py_DECREF(self);
        return NULL;
    }
    if (max_depth < 1 || max_depth > 21)
    {
        PyErr_SetString(PyExc_ValueError, "max_depth must be in [1, 21].");
        Py_DECREF(self);
        return NULL;
    }

    /* Discover the device, create ctx + queue. */
    if (!cl_resolve_device(device_name, &self->device))
    {
        Py_DECREF(self);
        return NULL;
    }
    cvl_cl_status_t status = cvl_cl_ctx_create(&self->device, &self->ctx);
    if (status != CVL_CL_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "failed to create OpenCL context: %s", cvl_cl_status_str(status));
        Py_DECREF(self);
        return NULL;
    }
    status = cvl_cl_queue_create(self->ctx, self->device.id, NULL, &self->queue);
    if (status != CVL_CL_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "failed to create OpenCL queue: %s", cvl_cl_status_str(status));
        cvl_cl_ctx_destroy(&self->ctx);
        Py_DECREF(self);
        return NULL;
    }

    self->precision = (strcmp(precision_name, "fp32") == 0) ? CVL_CL_PRECISION_FP32 : CVL_CL_PRECISION_FP64;
    self->n_threads = 1;
    self->closed = false;

    /* Compile the kernels. */
    status = cvl_cl_compute_init(&self->compute, self->ctx, self->queue, &self->device, self->precision, NULL, 0);
    if (status != CVL_CL_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "failed to compile OpenCL kernels: %s", cvl_cl_status_str(status));
        cvl_cl_queue_destroy(&self->queue);
        cvl_cl_ctx_destroy(&self->ctx);
        Py_DECREF(self);
        return NULL;
    }

    return (PyObject *)self;
}

static void pyvl_cl_backend_dealloc(PyObject *self)
{
    PyVL_CLBackendObject *this = (PyVL_CLBackendObject *)self;
    cvl_cl_compute_destroy(&this->compute);
    cvl_cl_queue_destroy(&this->queue);
    cvl_cl_ctx_destroy(&this->ctx);
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
}

static PyObject *pyvl_cl_backend_str(PyObject *self)
{
    const PyVL_CLBackendObject *this = (const PyVL_CLBackendObject *)self;
    char buf[256];
    snprintf(buf, sizeof(buf), "CLBackend(device=%s, %s)", this->device.info.name,
             this->precision == CVL_CL_PRECISION_FP32 ? "fp32" : "fp64");
    return PyUnicode_FromString(buf);
}

static PyObject *pyvl_cl_backend_repr(PyObject *self)
{
    const PyVL_CLBackendObject *this = (const PyVL_CLBackendObject *)self;
    char buf[320];
    snprintf(buf, sizeof(buf), "<CLBackend %s (%s) %s at %p>", this->device.info.name, this->device.info.vendor,
             this->precision == CVL_CL_PRECISION_FP32 ? "fp32" : "fp64", (void *)self);
    return PyUnicode_FromString(buf);
}

/* --- Properties --- */

static PyObject *pyvl_cl_backend_get_device_name(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLBackendObject *this = (const PyVL_CLBackendObject *)self;
    return PyUnicode_FromString(this->device.info.name);
}

static PyObject *pyvl_cl_backend_get_vendor(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLBackendObject *this = (const PyVL_CLBackendObject *)self;
    return PyUnicode_FromString(this->device.info.vendor);
}

static PyObject *pyvl_cl_backend_get_device_type(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLBackendObject *this = (const PyVL_CLBackendObject *)self;
    /* Not cached in cvl_cl_device_t; re-query. */
    cl_device_type dt = 0;
    clGetDeviceInfo(this->device.id, CL_DEVICE_TYPE, sizeof(dt), &dt, NULL);
    if (dt & CL_DEVICE_TYPE_GPU)
        return PyUnicode_FromString("gpu");
    if (dt & CL_DEVICE_TYPE_CPU)
        return PyUnicode_FromString("cpu");
    return PyUnicode_FromString("unknown");
}

static PyObject *pyvl_cl_backend_get_precision(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLBackendObject *this = (const PyVL_CLBackendObject *)self;
    return PyUnicode_FromString(this->precision == CVL_CL_PRECISION_FP32 ? "fp32" : "fp64");
}

static PyObject *pyvl_cl_backend_get_closed(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLBackendObject *this = (const PyVL_CLBackendObject *)self;
    return PyBool_FromLong(this->closed);
}

static PyGetSetDef pyvl_cl_backend_getset[] = {
    {
        .name = "device_name",
        .get = pyvl_cl_backend_get_device_name,
        .doc = "str : OpenCL device name.",
    },
    {
        .name = "vendor",
        .get = pyvl_cl_backend_get_vendor,
        .doc = "str : OpenCL device vendor.",
    },
    {
        .name = "device_type",
        .get = pyvl_cl_backend_get_device_type,
        .doc = "str : 'gpu', 'cpu', or 'unknown'.",
    },
    {
        .name = "precision",
        .get = pyvl_cl_backend_get_precision,
        .doc = "str : 'fp64' or 'fp32'.",
    },
    {
        .name = "closed",
        .get = pyvl_cl_backend_get_closed,
        .doc = "bool : True once close() has been called.",
    },
    {0},
};

/* --- Methods --- */

PyDoc_STRVAR(pyvl_cl_backend_build_tree_doc, "build_tree(coords, values)\n"
                                             "Enqueue a tree build and return a :class:`CLTreeBuild` future.\n"
                                             "\n"
                                             "Parameters\n"
                                             "----------\n"
                                             "coords : (..., 3) array_like\n"
                                             "    Source positions.\n"
                                             "values : (..., 3) array_like\n"
                                             "    Source strengths.\n"
                                             "\n"
                                             "Returns\n"
                                             "-------\n"
                                             "CLTreeBuild\n"
                                             "    Future whose ``.result()`` returns a populated :class:`CLTree`.\n"
                                             "\n"
                                             "See Also\n"
                                             "--------\n"
                                             "CLTreeBuild.result : Block and return the tree.\n");

static PyObject *pyvl_cl_backend_build_tree(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                            const Py_ssize_t nargs, const PyObject *kwnames)
{
    (void)defining_class;
    PyVL_CLBackendObject *backend = (PyVL_CLBackendObject *)self;

    PyObject *coords_obj, *values_obj;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&coords_obj},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&values_obj},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    size_t n_coords, n_vals;
    PyArrayObject *coords_arr = cl_flatten_points(coords_obj, "coords", &n_coords);
    if (!coords_arr)
        return NULL;
    PyArrayObject *values_arr = cl_flatten_points(values_obj, "values", &n_vals);
    if (!values_arr)
    {
        Py_DECREF(coords_arr);
        return NULL;
    }
    if (n_coords != n_vals)
    {
        PyErr_Format(PyExc_ValueError, "Number of coordinates (%zu) does not match values (%zu).", n_coords, n_vals);
        Py_DECREF(coords_arr);
        Py_DECREF(values_arr);
        return NULL;
    }

    /* Create the CLTree object via the module state type. */
    const module_state_t *state = get_module_state(Py_TYPE(self));
    if (!state)
    {
        Py_DECREF(coords_arr);
        Py_DECREF(values_arr);
        return NULL;
    }
    PyVL_CLTreeObject *tree_obj = (PyVL_CLTreeObject *)state->cl_tree_type->tp_alloc(state->cl_tree_type, 0);
    if (!tree_obj)
    {
        Py_DECREF(coords_arr);
        Py_DECREF(values_arr);
        return NULL;
    }
    tree_obj->backend = backend;
    Py_INCREF(backend);
    tree_obj->built = false;
    tree_obj->sources_coords = NULL;
    tree_obj->sources_values = NULL;
    tree_obj->n_threads = backend->n_threads;

    /* Init the embedded C tree. */
    const cvl_cl_flat_tree_settings_t settings = {
        .max_depth = 20,
        .critical_particle_count = 4,
        .order = 4,
    };
    cvl_cl_tree_init(&tree_obj->tree, &backend->compute, backend->precision, &settings, 0);
    tree_obj->tree.n_sources = (unsigned)n_coords;

    /* Store the sources. */
    tree_obj->sources_coords = (PyObject *)coords_arr;
    tree_obj->sources_values = (PyObject *)values_arr;

    /* Create the CLTreeBuild future. */
    PyVL_CLTreeBuildObject *build =
        (PyVL_CLTreeBuildObject *)state->cl_tree_build_type->tp_alloc(state->cl_tree_build_type, 0);
    if (!build)
    {
        Py_DECREF((PyObject *)tree_obj);
        return NULL;
    }
    build->tree = tree_obj;
    Py_INCREF(tree_obj);
    build->result = NULL;
    build->exception = NULL;
    build->done = false;
    /* The C job is unused for now (result() runs the sync build directly). */
    memset(&build->job, 0, sizeof(build->job));

    return (PyObject *)build;
}

PyDoc_STRVAR(pyvl_cl_backend_close_doc, "close()\n"
                                        "Release all device resources owned by the backend.\n"
                                        "\n"
                                        "Trees created from this backend become unusable.  The backend\n"
                                        "is also released when garbage-collected.\n");

static PyObject *pyvl_cl_backend_close(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyVL_CLBackendObject *this = (PyVL_CLBackendObject *)self;
    if (!this->closed)
    {
        this->closed = true;
        cvl_cl_compute_destroy(&this->compute);
        cvl_cl_queue_destroy(&this->queue);
        cvl_cl_ctx_destroy(&this->ctx);
    }
    Py_RETURN_NONE;
}

static PyMethodDef pyvl_cl_backend_methods[] = {
    {
        .ml_name = "build_tree",
        .ml_meth = (void *)pyvl_cl_backend_build_tree,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = pyvl_cl_backend_build_tree_doc,
    },
    {
        .ml_name = "close",
        .ml_meth = (void *)pyvl_cl_backend_close,
        .ml_flags = METH_NOARGS,
        .ml_doc = pyvl_cl_backend_close_doc,
    },
    {0},
};

PyType_Spec pyvl_cl_backend_typespec = {
    .name = PYVL_CTYPE_NAME(CLBackend),
    .basicsize = sizeof(PyVL_CLBackendObject),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_str, pyvl_cl_backend_str},
            {Py_tp_repr, pyvl_cl_backend_repr},
            {Py_tp_doc, (void *)pyvl_cl_backend_type_docstring},
            {Py_tp_methods, pyvl_cl_backend_methods},
            {Py_tp_getset, pyvl_cl_backend_getset},
            {Py_tp_new, pyvl_cl_backend_new},
            {Py_tp_dealloc, pyvl_cl_backend_dealloc},
            {0},
        },
};

/* Forward-declared in the header; the actual type objects live in the
 * module state (created from the specs below). */

/* ------------------------------------------------------------------ */
/*  CLTree                                                             */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(pyvl_cl_tree_type_docstring, "CLTree(order=4, critical_particle_count=4, max_depth=20)\n"
                                          "OpenCL-backed octree for far-field induction.\n"
                                          "\n"
                                          "A :class:`CLTree` is created by :meth:`CLBackend.build_tree` and\n"
                                          "owns device-side tree buffers that are reused across rebuilds.\n"
                                          "\n"
                                          "See Also\n"
                                          "--------\n"
                                          "CLBackend.build_tree : Create a tree.\n"
                                          "CLTree.eval : Evaluate the tree at target points.\n");

static void pyvl_cl_tree_dealloc(PyObject *self)
{
    PyVL_CLTreeObject *this = (PyVL_CLTreeObject *)self;
    cvl_cl_tree_destroy(&this->tree);
    Py_XDECREF(this->sources_coords);
    Py_XDECREF(this->sources_values);
    Py_XDECREF((PyObject *)this->backend);
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
}

static int pyvl_cl_tree_traverse(PyObject *self, visitproc visit, void *arg)
{
    PyVL_CLTreeObject *this = (PyVL_CLTreeObject *)self;
    Py_VISIT(this->sources_coords);
    Py_VISIT(this->sources_values);
    Py_VISIT((PyObject *)this->backend);
    return 0;
}

static PyObject *pyvl_cl_tree_str(PyObject *self)
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    if (this->built)
    {
        char buf[160];
        snprintf(buf, sizeof(buf), "CLTree(n_sources=%u, n_nodes=%u, n_internal=%u, n_leaves=%u)", this->tree.n_sources,
                 this->tree.n_nodes, this->tree.n_internal, this->tree.n_leaves);
        return PyUnicode_FromString(buf);
    }
    return PyUnicode_FromString("CLTree(unbuilt)");
}

static PyObject *pyvl_cl_tree_repr(PyObject *self)
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    if (this->built)
    {
        char buf[200];
        snprintf(buf, sizeof(buf), "<CLTree n_sources=%u n_nodes=%u n_internal=%u n_leaves=%u at %p>",
                 this->tree.n_sources, this->tree.n_nodes, this->tree.n_internal, this->tree.n_leaves, (void *)self);
        return PyUnicode_FromString(buf);
    }
    return PyUnicode_FromString("<CLTree unbuilt>");
}

/* --- Properties --- */

static PyObject *pyvl_cl_tree_get_n_sources(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_sources);
}

static PyObject *pyvl_cl_tree_get_n_nodes(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_nodes);
}

static PyObject *pyvl_cl_tree_get_n_internal(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_internal);
}

static PyObject *pyvl_cl_tree_get_n_multipole_leaves(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_multipole_leaves);
}

static PyObject *pyvl_cl_tree_get_n_particle_leaves(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_particle_leaves);
}

static PyObject *pyvl_cl_tree_get_n_leaves(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_leaves);
}

static PyObject *pyvl_cl_tree_get_max_depth(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    return PyLong_FromUnsignedLong(this->tree.max_depth_used);
}

static PyObject *pyvl_cl_tree_get_order(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    return PyLong_FromUnsignedLong(this->tree.settings.order);
}

static PyObject *pyvl_cl_tree_get_built(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_CLTreeObject *this = (const PyVL_CLTreeObject *)self;
    return PyBool_FromLong(this->built);
}

static PyGetSetDef pyvl_cl_tree_getset[] = {
    {
        .name = "n_sources",
        .get = pyvl_cl_tree_get_n_sources,
        .doc = "int : Number of source particles.",
    },
    {
        .name = "n_nodes",
        .get = pyvl_cl_tree_get_n_nodes,
        .doc = "int : Total number of octree nodes.",
    },
    {
        .name = "n_internal",
        .get = pyvl_cl_tree_get_n_internal,
        .doc = "int : Number of internal (non-leaf) nodes.",
    },
    {
        .name = "n_multipole_leaves",
        .get = pyvl_cl_tree_get_n_multipole_leaves,
        .doc = "int : Number of multipole-bearing leaves.",
    },
    {
        .name = "n_particle_leaves",
        .get = pyvl_cl_tree_get_n_particle_leaves,
        .doc = "int : Number of uncompressed particle leaves.",
    },
    {
        .name = "n_leaves",
        .get = pyvl_cl_tree_get_n_leaves,
        .doc = "int : Number of leaves.",
    },
    {
        .name = "max_depth",
        .get = pyvl_cl_tree_get_max_depth,
        .doc = "int : Maximum depth actually reached.",
    },
    {
        .name = "order",
        .get = pyvl_cl_tree_get_order,
        .doc = "int : Multipole expansion order.",
    },
    {
        .name = "built",
        .get = pyvl_cl_tree_get_built,
        .doc = "bool : True once the tree has been built.",
    },
    {0},
};

/* --- Methods --- */

PyDoc_STRVAR(pyvl_cl_tree_eval_doc, "eval(targets, /, *, mode='tree_code', theta=0.0)\n"
                                    "Enqueue an evaluation and return a :class:`CLTreeEval` future.\n"
                                    "\n"
                                    "Parameters\n"
                                    "----------\n"
                                    "targets : (..., 3) array_like\n"
                                    "    Points at which to evaluate.\n"
                                    "mode : str, default 'tree_code'\n"
                                    "    ``'tree_code'`` -- multipole tree-code (MAC-driven).\n"
                                    "    ``'direct'`` -- exact O(N) direct sum per target.\n"
                                    "theta : float, default 0.0\n"
                                    "    MAC opening angle (``<= 0`` uses the neighbour criterion).\n"
                                    "\n"
                                    "Returns\n"
                                    "-------\n"
                                    "CLTreeEval\n"
                                    "    Future whose ``.result()`` returns the (..., 3) ndarray.\n"
                                    "\n"
                                    "See Also\n"
                                    "--------\n"
                                    "CLTreeEval.result : Block and return the induced field.\n");

static PyObject *pyvl_cl_tree_eval(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                   const Py_ssize_t nargs, const PyObject *kwnames)
{
    (void)defining_class;
    PyVL_CLTreeObject *this = (PyVL_CLTreeObject *)self;

    PyObject *targets_obj;
    PyObject *mode_obj = NULL;
    double theta = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&targets_obj},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&mode_obj,
                 .kwname = "mode",
                 .optional = true,
                 .kw_only = true},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &theta, .kwname = "theta", .optional = true, .kw_only = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    cvl_cl_tree_eval_mode_t mode = CVL_CL_TREE_EVAL_TREE_CODE;
    if (mode_obj != NULL && !Py_IsNone(mode_obj))
    {
        if (!PyUnicode_Check(mode_obj))
        {
            PyErr_SetString(PyExc_TypeError, "mode must be a string ('tree_code' or 'direct').");
            return NULL;
        }
        const char *mode_str = PyUnicode_AsUTF8(mode_obj);
        if (strcmp(mode_str, "tree_code") == 0)
            mode = CVL_CL_TREE_EVAL_TREE_CODE;
        else if (strcmp(mode_str, "direct") == 0)
            mode = CVL_CL_TREE_EVAL_DIRECT;
        else
        {
            PyErr_Format(PyExc_ValueError, "Unknown mode \"%s\". Must be \"tree_code\" or \"direct\".", mode_str);
            return NULL;
        }
    }

    size_t n_targets;
    PyArrayObject *targets_arr = cl_flatten_points(targets_obj, "targets", &n_targets);
    if (!targets_arr)
        return NULL;

    const int ndim = PyArray_NDIM(targets_arr);
    const npy_intp *dims = PyArray_DIMS(targets_arr);

    /* Create the eval future object via the module state type. */
    const module_state_t *state = get_module_state(Py_TYPE(self));
    if (!state)
    {
        Py_DECREF(targets_arr);
        return NULL;
    }
    PyVL_CLTreeEvalObject *job =
        (PyVL_CLTreeEvalObject *)state->cl_tree_eval_type->tp_alloc(state->cl_tree_eval_type, 0);
    if (!job)
    {
        Py_DECREF(targets_arr);
        return NULL;
    }
    job->tree = this;
    Py_INCREF(this);
    job->result = NULL;
    job->exception = NULL;
    job->done = false;
    job->host_out = NULL;
    job->scratch_f32 = NULL;
    job->n_targets = n_targets;
    job->ndim = ndim;
    job->dims_arr = (Py_ssize_t *)malloc((size_t)ndim * sizeof(Py_ssize_t));
    if (!job->dims_arr)
    {
        Py_DECREF((PyObject *)job);
        Py_DECREF(targets_arr);
        return NULL;
    }
    for (int i = 0; i < ndim; ++i)
        job->dims_arr[i] = dims[i];

    const real3_t *targets_data = (const real3_t *)PyArray_DATA(targets_arr);
    cvl_cl_status_t status =
        cvl_cl_tree_eval_begin(&this->tree, (unsigned)n_targets, targets_data, mode, theta, &job->job);
    Py_DECREF(targets_arr);
    if (status != CVL_CL_SUCCESS)
    {
        Py_DECREF((PyObject *)job);
        return cl_raise_status(status, "eval");
    }

    return (PyObject *)job;
}

PyDoc_STRVAR(pyvl_cl_tree_eval_sources_doc, "eval_sources(/, *, mode='tree_code', theta=0.0)\n"
                                            "Enqueue an evaluation at the source positions and return a\n"
                                            ":class:`CLTreeEval` future.\n"
                                            "\n"
                                            "Equivalent to ``eval(sources_coords)`` but avoids re-uploading\n"
                                            "the source positions (they are already on the device).\n"
                                            "\n"
                                            "Parameters\n"
                                            "----------\n"
                                            "mode : str, default 'tree_code'\n"
                                            "    ``'tree_code'`` or ``'direct'``.\n"
                                            "theta : float, default 0.0\n"
                                            "    MAC opening angle.\n"
                                            "\n"
                                            "Returns\n"
                                            "-------\n"
                                            "CLTreeEval\n"
                                            "    Future whose ``.result()`` returns the (..., 3) ndarray.\n");

static PyObject *pyvl_cl_tree_eval_sources(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                           const Py_ssize_t nargs, const PyObject *kwnames)
{
    (void)defining_class;
    (void)args;
    (void)nargs;
    (void)kwnames;
    PyVL_CLTreeObject *this = (PyVL_CLTreeObject *)self;

    PyObject *mode_obj = NULL;
    double theta = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&mode_obj,
                 .kwname = "mode",
                 .optional = true,
                 .kw_only = true},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &theta, .kwname = "theta", .optional = true, .kw_only = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (!this->built || !this->sources_coords)
    {
        PyErr_SetString(PyExc_RuntimeError, "CLTree has not been built. Call result() on the build future first.");
        return NULL;
    }

    /* Use the stored source coordinates as the targets (already on device). */
    PyArrayObject *targets_arr = (PyArrayObject *)this->sources_coords;
    const size_t n_targets = this->tree.n_sources;
    const int ndim = PyArray_NDIM(targets_arr);
    const npy_intp *dims = PyArray_DIMS(targets_arr);

    cvl_cl_tree_eval_mode_t mode = CVL_CL_TREE_EVAL_TREE_CODE;
    if (mode_obj != NULL && !Py_IsNone(mode_obj))
    {
        if (!PyUnicode_Check(mode_obj))
        {
            PyErr_SetString(PyExc_TypeError, "mode must be a string ('tree_code' or 'direct').");
            return NULL;
        }
        const char *mode_str = PyUnicode_AsUTF8(mode_obj);
        if (strcmp(mode_str, "tree_code") == 0)
            mode = CVL_CL_TREE_EVAL_TREE_CODE;
        else if (strcmp(mode_str, "direct") == 0)
            mode = CVL_CL_TREE_EVAL_DIRECT;
        else
        {
            PyErr_Format(PyExc_ValueError, "Unknown mode \"%s\". Must be \"tree_code\" or \"direct\".", mode_str);
            return NULL;
        }
    }

    /* Create the eval future object. */
    const module_state_t *state = get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    PyVL_CLTreeEvalObject *job =
        (PyVL_CLTreeEvalObject *)state->cl_tree_eval_type->tp_alloc(state->cl_tree_eval_type, 0);
    if (!job)
        return NULL;
    job->tree = this;
    Py_INCREF(this);
    job->result = NULL;
    job->exception = NULL;
    job->done = false;
    job->host_out = NULL;
    job->scratch_f32 = NULL;
    job->n_targets = n_targets;
    job->ndim = ndim;
    job->dims_arr = (Py_ssize_t *)malloc((size_t)ndim * sizeof(Py_ssize_t));
    if (!job->dims_arr)
    {
        Py_DECREF((PyObject *)job);
        return NULL;
    }
    for (int i = 0; i < ndim; ++i)
        job->dims_arr[i] = dims[i];

    const real3_t *targets_data = (const real3_t *)PyArray_DATA(targets_arr);
    cvl_cl_status_t status =
        cvl_cl_tree_eval_begin(&this->tree, (unsigned)n_targets, targets_data, mode, theta, &job->job);
    if (status != CVL_CL_SUCCESS)
    {
        Py_DECREF((PyObject *)job);
        return cl_raise_status(status, "eval_sources");
    }

    return (PyObject *)job;
}

PyDoc_STRVAR(pyvl_cl_tree_rebuild_doc, "rebuild(coords, values)\n"
                                       "Enqueue a rebuild of this tree and return a :class:`CLTreeBuild`\n"
                                       "future.\n"
                                       "\n"
                                       "The tree's device buffers are reused (grow-only) so repeated\n"
                                       "rebuilds avoid re-allocating.  The returned future's\n"
                                       "``.result()`` returns this same tree, rebuilt.\n"
                                       "\n"
                                       "Parameters\n"
                                       "----------\n"
                                       "coords : (..., 3) array_like\n"
                                       "    New source positions.\n"
                                       "values : (..., 3) array_like\n"
                                       "    New source strengths.\n"
                                       "\n"
                                       "Returns\n"
                                       "-------\n"
                                       "CLTreeBuild\n"
                                       "    Future; ``.result()`` returns the rebuilt :class:`CLTree`.\n");

static PyObject *pyvl_cl_tree_rebuild(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                      const Py_ssize_t nargs, const PyObject *kwnames)
{
    (void)defining_class;
    PyVL_CLTreeObject *this = (PyVL_CLTreeObject *)self;

    PyObject *coords_obj, *values_obj;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&coords_obj},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&values_obj},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    size_t n_coords, n_vals;
    PyArrayObject *coords_arr = cl_flatten_points(coords_obj, "coords", &n_coords);
    if (!coords_arr)
        return NULL;
    PyArrayObject *values_arr = cl_flatten_points(values_obj, "values", &n_vals);
    if (!values_arr)
    {
        Py_DECREF(coords_arr);
        return NULL;
    }
    if (n_coords != n_vals)
    {
        PyErr_Format(PyExc_ValueError, "Number of coordinates (%zu) does not match values (%zu).", n_coords, n_vals);
        Py_DECREF(coords_arr);
        Py_DECREF(values_arr);
        return NULL;
    }

    /* Replace the stored source arrays. */
    Py_XDECREF(this->sources_coords);
    Py_XDECREF(this->sources_values);
    this->sources_coords = (PyObject *)coords_arr;
    this->sources_values = (PyObject *)values_arr;
    this->tree.n_sources = (unsigned)n_coords;
    this->built = false;

    /* Create the build future. */
    const module_state_t *state = get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    PyVL_CLTreeBuildObject *build =
        (PyVL_CLTreeBuildObject *)state->cl_tree_build_type->tp_alloc(state->cl_tree_build_type, 0);
    if (!build)
        return NULL;
    build->tree = this;
    Py_INCREF(this);
    build->result = NULL;
    build->exception = NULL;
    build->done = false;
    memset(&build->job, 0, sizeof(build->job));

    return (PyObject *)build;
}

static PyMethodDef pyvl_cl_tree_methods_full[] = {
    {
        .ml_name = "eval",
        .ml_meth = (void *)pyvl_cl_tree_eval,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = pyvl_cl_tree_eval_doc,
    },
    {
        .ml_name = "eval_sources",
        .ml_meth = (void *)pyvl_cl_tree_eval_sources,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = pyvl_cl_tree_eval_sources_doc,
    },
    {
        .ml_name = "rebuild",
        .ml_meth = (void *)pyvl_cl_tree_rebuild,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = pyvl_cl_tree_rebuild_doc,
    },
    {0},
};

PyType_Spec pyvl_cl_tree_typespec = {
    .name = PYVL_CTYPE_NAME(CLTree),
    .basicsize = sizeof(PyVL_CLTreeObject),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_str, pyvl_cl_tree_str},
            {Py_tp_repr, pyvl_cl_tree_repr},
            {Py_tp_doc, (void *)pyvl_cl_tree_type_docstring},
            {Py_tp_methods, pyvl_cl_tree_methods_full},
            {Py_tp_getset, pyvl_cl_tree_getset},
            {Py_tp_dealloc, pyvl_cl_tree_dealloc},
            {Py_tp_traverse, pyvl_cl_tree_traverse},
            {0},
        },
};

/* ------------------------------------------------------------------ */
/*  Futures (CLTreeBuild / CLTreeEval)                                 */
/* ------------------------------------------------------------------ */

PyDoc_STRVAR(pyvl_cl_tree_build_type_docstring, "CLTreeBuild()\n"
                                                "Future returned by :meth:`CLBackend.build_tree`.\n"
                                                "\n"
                                                "Call ``.result()`` to block until the build finishes and obtain\n"
                                                "the populated :class:`CLTree`.\n");

static void pyvl_cl_tree_build_dealloc(PyObject *self)
{
    PyVL_CLTreeBuildObject *this = (PyVL_CLTreeBuildObject *)self;
    Py_XDECREF(this->result);
    Py_XDECREF(this->exception);
    Py_XDECREF((PyObject *)this->tree);
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
}

static int pyvl_cl_tree_build_traverse(PyObject *self, visitproc visit, void *arg)
{
    PyVL_CLTreeBuildObject *this = (PyVL_CLTreeBuildObject *)self;
    Py_VISIT(this->result);
    Py_VISIT(this->exception);
    Py_VISIT((PyObject *)this->tree);
    return 0;
}

static PyObject *pyvl_cl_tree_build_done(PyObject *self, PyObject *Py_UNUSED(args))
{
    const PyVL_CLTreeBuildObject *this = (const PyVL_CLTreeBuildObject *)self;
    return PyBool_FromLong(this->done);
}

PyDoc_STRVAR(pyvl_cl_tree_build_result_doc, "result()\n"
                                            "Block until the build completes and return the :class:`CLTree`.\n"
                                            "\n"
                                            "Raises the build error (if any).\n");

static PyObject *pyvl_cl_tree_build_result(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyVL_CLTreeBuildObject *this = (PyVL_CLTreeBuildObject *)self;
    if (this->done)
    {
        if (this->exception)
        {
            PyErr_SetObject(PyExc_RuntimeError, this->exception);
            return NULL;
        }
        Py_INCREF(this->result);
        return this->result;
    }

    /* Block: run the host-side build steps via the sync build on the
     * stored source arrays. */
    PyVL_CLTreeObject *tree = this->tree;
    const real3_t *coords = (const real3_t *)PyArray_DATA((PyArrayObject *)tree->sources_coords);
    const real3_t *values = (const real3_t *)PyArray_DATA((PyArrayObject *)tree->sources_values);
    const unsigned n_sources = tree->tree.n_sources;
    const size_t work_sz = cvl_cl_tree_build_work_size(n_sources, tree->tree.settings.max_depth);
    void *work = PyMem_Malloc(work_sz);
    if (!work)
    {
        PyErr_NoMemory();
        return NULL;
    }

    cvl_cl_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    status = cvl_cl_tree_build(&tree->tree, n_sources, coords, values, work, work_sz);
    Py_END_ALLOW_THREADS;
    PyMem_Free(work);

    this->done = true;
    if (status != CVL_CL_SUCCESS)
    {
        this->exception = PyUnicode_FromString(cvl_cl_status_str(status));
        PyErr_Format(PyExc_RuntimeError, "tree build failed: %s", cvl_cl_status_str(status));
        return NULL;
    }

    /* Mark the tree built. */
    tree->built = true;
    this->result = (PyObject *)tree;
    Py_INCREF(tree);
    return (PyObject *)tree;
}

static PyMethodDef pyvl_cl_tree_build_methods[] = {
    {
        .ml_name = "done",
        .ml_meth = pyvl_cl_tree_build_done,
        .ml_flags = METH_NOARGS,
        .ml_doc = "done()\nReturn True if the build has finished.",
    },
    {
        .ml_name = "result",
        .ml_meth = pyvl_cl_tree_build_result,
        .ml_flags = METH_NOARGS,
        .ml_doc = pyvl_cl_tree_build_result_doc,
    },
    {0},
};

PyType_Spec pyvl_cl_tree_build_typespec = {
    .name = PYVL_CTYPE_NAME(CLTreeBuild),
    .basicsize = sizeof(PyVL_CLTreeBuildObject),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_doc, (void *)pyvl_cl_tree_build_type_docstring},
            {Py_tp_methods, pyvl_cl_tree_build_methods},
            {Py_tp_dealloc, pyvl_cl_tree_build_dealloc},
            {Py_tp_traverse, pyvl_cl_tree_build_traverse},
            {0},
        },
};

PyDoc_STRVAR(pyvl_cl_tree_eval_type_docstring, "CLTreeEval()\n"
                                               "Future returned by :meth:`CLTree.eval`.\n"
                                               "\n"
                                               "Call ``.result()`` to block until the evaluation finishes and\n"
                                               "obtain the (..., 3) ndarray of induced fields.\n");

static void pyvl_cl_tree_eval_dealloc(PyObject *self)
{
    PyVL_CLTreeEvalObject *this = (PyVL_CLTreeEvalObject *)self;
    if (!this->done)
        cvl_cl_tree_eval_cancel(&this->job);
    free(this->host_out);
    free(this->scratch_f32);
    free(this->dims_arr);
    Py_XDECREF(this->result);
    Py_XDECREF(this->exception);
    Py_XDECREF((PyObject *)this->tree);
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
}

static int pyvl_cl_tree_eval_traverse(PyObject *self, visitproc visit, void *arg)
{
    PyVL_CLTreeEvalObject *this = (PyVL_CLTreeEvalObject *)self;
    Py_VISIT(this->result);
    Py_VISIT(this->exception);
    Py_VISIT((PyObject *)this->tree);
    return 0;
}

static PyObject *pyvl_cl_tree_eval_done(PyObject *self, PyObject *Py_UNUSED(args))
{
    const PyVL_CLTreeEvalObject *this = (const PyVL_CLTreeEvalObject *)self;
    return PyBool_FromLong(this->done);
}

PyDoc_STRVAR(pyvl_cl_tree_eval_result_doc, "result()\n"
                                           "Block until the evaluation completes and return the (..., 3) ndarray.\n"
                                           "\n"
                                           "Raises the evaluation error (if any).\n");

static PyObject *pyvl_cl_tree_eval_result(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyVL_CLTreeEvalObject *this = (PyVL_CLTreeEvalObject *)self;
    if (this->done)
    {
        if (this->exception)
        {
            PyErr_SetObject(PyExc_RuntimeError, this->exception);
            return NULL;
        }
        Py_INCREF(this->result);
        return this->result;
    }

    /* Allocate host readback scratch. */
    const size_t n_targets = this->n_targets;
    this->host_out = malloc((size_t)n_targets * sizeof(real3_t));
    if (!this->host_out)
    {
        PyErr_NoMemory();
        return NULL;
    }

    cvl_cl_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    status = cvl_cl_tree_eval_finish(&this->job, (real3_t *)this->host_out, NULL);
    Py_END_ALLOW_THREADS;

    this->done = true;
    if (status != CVL_CL_SUCCESS)
    {
        this->exception = PyUnicode_FromString(cvl_cl_status_str(status));
        PyErr_Format(PyExc_RuntimeError, "tree eval failed: %s", cvl_cl_status_str(status));
        return NULL;
    }

    /* Build the output ndarray. */
    Py_ssize_t *dims = (Py_ssize_t *)malloc((size_t)this->ndim * sizeof(Py_ssize_t));
    if (!dims)
    {
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < this->ndim; ++i)
        dims[i] = this->dims_arr[i];
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(this->ndim, dims, NPY_DOUBLE);
    free(dims);
    if (!out)
        return NULL;

    memcpy(PyArray_DATA(out), this->host_out, (size_t)n_targets * 3u * sizeof(double));
    this->result = (PyObject *)out;
    Py_INCREF(out);
    return (PyObject *)out;
}

static PyMethodDef pyvl_cl_tree_eval_methods[] = {
    {
        .ml_name = "done",
        .ml_meth = pyvl_cl_tree_eval_done,
        .ml_flags = METH_NOARGS,
        .ml_doc = "done()\nReturn True if the evaluation has finished.",
    },
    {
        .ml_name = "result",
        .ml_meth = pyvl_cl_tree_eval_result,
        .ml_flags = METH_NOARGS,
        .ml_doc = pyvl_cl_tree_eval_result_doc,
    },
    {0},
};

PyType_Spec pyvl_cl_tree_eval_typespec = {
    .name = PYVL_CTYPE_NAME(CLTreeEval),
    .basicsize = sizeof(PyVL_CLTreeEvalObject),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_doc, (void *)pyvl_cl_tree_eval_type_docstring},
            {Py_tp_methods, pyvl_cl_tree_eval_methods},
            {Py_tp_dealloc, pyvl_cl_tree_eval_dealloc},
            {Py_tp_traverse, pyvl_cl_tree_eval_traverse},
            {0},
        },
};
