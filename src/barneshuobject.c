#include "barneshuobject.h"
#include "allocator.h"
#include "core/common.h"
#include "core/cost_model.h"
#include <numpy/arrayobject.h>
// this goes last
#include <cpyutl.h>

/*
 * ----------------------------------------------------------------
 *  Helpers
 * ----------------------------------------------------------------
 */

/**
 * @brief Type-check self and retrieve module state (bound-method helper).
 */
static bool ensure_bh_and_state(PyTypeObject *defining_class, PyObject *self, const PyVL_BarnesHutObject **p_this,
                                const module_state_t **p_state)
{
    const module_state_t *state = NULL;
    if (defining_class)
        state = PyType_GetModuleState(defining_class);
    else
        state = get_module_state(Py_TYPE(self));

    if (!state)
        return false;

    if (!PyObject_TypeCheck(self, state->bh_tree_type))
    {
        PyErr_Format(PyExc_TypeError, "Self was not \"%s\" but was \"%s\".", state->bh_tree_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return false;
    }
    *p_this = (const PyVL_BarnesHutObject *)self;
    *p_state = state;
    return true;
}

/**
 * @brief Convert a (..., 3) array-like to a contiguous double array and
 *        return the number of "points" (product of leading dims).
 *
 * The caller owns the returned PyArrayObject reference. On failure NULL is
 * returned and a Python exception is set.
 */
static PyArrayObject *bh_flatten_points(PyObject *obj, const char *arg_name, size_t *p_n)
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
 * @brief Resolve the Python-level work_order to the C-level value.
 *
 * Python: `None` (encoded as work_order_py < 0) → 0 (meaning "use order").
 * Python: any non-negative value → pass through.
 */
static inline unsigned resolve_py_work_order(Py_ssize_t work_order_py)
{
    return (work_order_py < 0) ? 0u : (unsigned)work_order_py;
}

/*
 * ----------------------------------------------------------------
 *  tp_new / tp_init / tp_dealloc / tp_traverse / tp_str / tp_repr
 * ----------------------------------------------------------------
 */

PyDoc_STRVAR(pyvl_bh_tree_type_docstring,
             "BarnesHutTree(order=4, critical_particle_count=4, max_depth=20, work_order=None, "
             "alpha_centroid=0.5)\n"
             "Barnes-Hut octree for fast far-field induction from vortex particles.\n"
             "\n"
             "The tree partitions a set of vortex sources into an octree and\n"
             "compresses distant leaves into multipole expansions.  This replaces the\n"
             "direct :math:`O(N^2)` summation with a far-field :math:`O(N \\log N)`\n"
             "approximation.\n"
             "\n"
             "Use the classmethod ``build()`` to construct a populated tree from source\n"
             "arrays:\n"
             "\n"
             ">>> import numpy as np\n"
             ">>> from pyvl.cvl import BarnesHutTree\n"
             ">>> rng = np.random.default_rng(42)\n"
             ">>> coords = rng.uniform(-1, 1, (100, 3))\n"
             ">>> values = rng.uniform(-1, 1, (100, 3))\n"
             ">>> tree = BarnesHutTree.build(coords, values, order=4)\n"
             ">>> tree.eval(np.array([[10., 0., 0.], [0., 10., 0.]]))\n"
             "array([[ ...\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "order : int, default 4\n"
             "    Multipole order used by every compressed leaf.\n"
             "critical_particle_count : int, default 4\n"
             "    Base subdivision threshold.\n"
             "max_depth : int, default 20\n"
             "    Maximum octree depth.\n"
             "work_order : int or None, default None\n"
             "    Internal expansion order for multipole shifting (``None`` = use *order*).\n"
             "alpha_centroid : float, default 0.5\n"
             "    Centroid-based subdivision threshold (0.0 = disabled).  When > 0, a leaf is\n"
             "    subdivided if any source is more than ``alpha_centroid * half_size`` from\n"
             "    the geometric center, guaranteeing tightly clustered sources in each cell.\n"
             "\n"
             "See Also\n"
             "--------\n"
             "BarnesHutTree.build : Construct and populate in one step.\n"
             "\n"
             "Examples\n"
             "--------\n"
             ">>> from pyvl.cvl import BarnesHutTree\n"
             ">>> # Empty tree (settings only); use build() to populate.\n"
             ">>> t = BarnesHutTree(order=4)\n");

static PyObject *pyvl_bh_tree_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyVL_BarnesHutObject *self = (PyVL_BarnesHutObject *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    // Zero-initialise everything.
    self->tree = (barnes_hut_tree_t){0};
    self->n_threads = 1;
    self->sources_coords = NULL;
    self->sources_values = NULL;
    self->built = false;

    Py_ssize_t order = 4, critical = 4, max_depth = 20, work_order = -1;
    double alpha_centroid = 0.5;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|nnnnd",
            (char *[6]){"order", "critical_particle_count", "max_depth", "work_order", "alpha_centroid", NULL}, &order,
            &critical, &max_depth, &work_order, &alpha_centroid))
    {
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
    if (max_depth < 1)
    {
        PyErr_SetString(PyExc_ValueError, "max_depth must be >= 1.");
        Py_DECREF(self);
        return NULL;
    }

    if (alpha_centroid < 0.0)
    {
        PyErr_SetString(PyExc_ValueError, "alpha_centroid must be >= 0.");
        Py_DECREF(self);
        return NULL;
    }

    self->tree.settings = (barnes_hut_settings_t){
        .order = (unsigned)order,
        .critical_particle_count = (unsigned)critical,
        .max_depth = (unsigned)max_depth,
        .work_order = resolve_py_work_order(work_order),
        .alpha_centroid = (real_t)alpha_centroid,
    };

    return (PyObject *)self;
}

static void pyvl_bh_tree_dealloc(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    PyVL_BarnesHutObject *this = (PyVL_BarnesHutObject *)self;

    // Free the tree buffer (allocated via CVL_MEM_ALLOCATOR → PyMem_Malloc).
    if (this->tree.buffer)
    {
        CVL_MEM_ALLOCATOR.deallocate(CVL_MEM_ALLOCATOR.state, this->tree.buffer);
        this->tree.buffer = NULL;
    }

    Py_XDECREF(this->sources_coords);
    Py_XDECREF(this->sources_values);

    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
}

static int pyvl_bh_tree_traverse(PyObject *self, visitproc visit, void *arg)
{
    PyVL_BarnesHutObject *this = (PyVL_BarnesHutObject *)self;
    Py_VISIT(this->sources_coords);
    Py_VISIT(this->sources_values);
    Py_VISIT(Py_TYPE(self));
    return 0;
}

static PyObject *pyvl_bh_tree_str(PyObject *self)
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    if (this->built)
    {
        char buf[160];
        snprintf(buf, sizeof(buf),
                 "BarnesHutTree(order=%u, n_sources=%u, n_nodes=%u, "
                 "depth=%u, int=%u, mp=%u, ptcl=%u)",
                 this->tree.settings.order, this->tree.n_sources, this->tree.n_nodes, this->tree.max_depth_reached,
                 this->tree.n_internal, this->tree.n_multipole_leaves, this->tree.n_particle_leaves);
        return PyUnicode_FromString(buf);
    }
    else
    {
        char buf[80];
        snprintf(buf, sizeof(buf), "BarnesHutTree(order=%u, unbuilt)", this->tree.settings.order);
        return PyUnicode_FromString(buf);
    }
}

static PyObject *pyvl_bh_tree_repr(PyObject *self)
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    if (this->built)
    {
        char buf[200];
        snprintf(buf, sizeof(buf), "<BarnesHutTree order=%u n_sources=%u n_nodes=%u depth=%u at %p>",
                 this->tree.settings.order, this->tree.n_sources, this->tree.n_nodes, this->tree.max_depth_reached,
                 (void *)self);
        return PyUnicode_FromString(buf);
    }
    else
    {
        char buf[120];
        snprintf(buf, sizeof(buf), "<BarnesHutTree order=%u unbuilt at %p>", this->tree.settings.order, (void *)self);
        return PyUnicode_FromString(buf);
    }
}

/*
 * ----------------------------------------------------------------
 *  Properties
 * ----------------------------------------------------------------
 */

static PyObject *pyvl_bh_tree_get_n_sources(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_sources);
}

static PyObject *pyvl_bh_tree_get_n_nodes(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_nodes);
}

static PyObject *pyvl_bh_tree_get_n_internal(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_internal);
}

static PyObject *pyvl_bh_tree_get_n_multipole_leaves(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_multipole_leaves);
}

static PyObject *pyvl_bh_tree_get_n_particle_leaves(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyLong_FromUnsignedLong(this->tree.n_particle_leaves);
}

static PyObject *pyvl_bh_tree_get_max_depth(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyLong_FromUnsignedLong(this->tree.max_depth_reached);
}

static PyObject *pyvl_bh_tree_get_memory_bytes(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyLong_FromUnsignedLong((unsigned long)this->tree.buffer_size);
}

static PyObject *pyvl_bh_tree_get_order(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyLong_FromUnsignedLong(this->tree.settings.order);
}

static PyObject *pyvl_bh_tree_get_alpha_centroid(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyFloat_FromDouble((double)this->tree.settings.alpha_centroid);
}

static PyObject *pyvl_bh_tree_get_critical_particle_count(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyLong_FromUnsignedLong(this->tree.settings.critical_particle_count);
}

static PyObject *pyvl_bh_tree_get_max_depth_setting(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_BarnesHutObject *this = (const PyVL_BarnesHutObject *)self;
    return PyLong_FromUnsignedLong(this->tree.settings.max_depth);
}

static PyGetSetDef pyvl_bh_tree_getset[] = {
    {
        .name = "n_sources",
        .get = pyvl_bh_tree_get_n_sources,
        .doc = "int : Number of source particles.",
    },
    {
        .name = "n_nodes",
        .get = pyvl_bh_tree_get_n_nodes,
        .doc = "int : Total number of octree nodes.",
    },
    {
        .name = "n_internal",
        .get = pyvl_bh_tree_get_n_internal,
        .doc = "int : Number of internal (non-leaf) nodes.",
    },
    {
        .name = "n_multipole_leaves",
        .get = pyvl_bh_tree_get_n_multipole_leaves,
        .doc = "int : Number of multipole-bearing leaves.",
    },
    {
        .name = "n_particle_leaves",
        .get = pyvl_bh_tree_get_n_particle_leaves,
        .doc = "int : Number of uncompressed particle leaves.",
    },
    {
        .name = "max_depth",
        .get = pyvl_bh_tree_get_max_depth,
        .doc = "int : Maximum depth actually reached in the tree.",
    },
    {
        .name = "memory_bytes",
        .get = pyvl_bh_tree_get_memory_bytes,
        .doc = "int : Size of the tree buffer in bytes.",
    },
    {
        .name = "order",
        .get = pyvl_bh_tree_get_order,
        .doc = "int : Multipole order used by the tree.",
    },
    {
        .name = "alpha_centroid",
        .get = pyvl_bh_tree_get_alpha_centroid,
        .doc = "float : Centroid-based subdivision threshold (0.0 = disabled).",
    },
    {
        .name = "critical_particle_count",
        .get = pyvl_bh_tree_get_critical_particle_count,
        .doc = "int : Base subdivision threshold.",
    },
    {
        .name = "max_depth_setting",
        .get = pyvl_bh_tree_get_max_depth_setting,
        .doc = "int : Maximum depth cap configured at build time.",
    },
    {0},
};

/*
 * ----------------------------------------------------------------
 *  build(cls, sources_coords, sources_values, order=4,
 *        critical_particle_count=4, max_depth=20, work_order=None,
 *        n_threads=1)
 * ----------------------------------------------------------------
 */

PyDoc_STRVAR(pyvl_bh_tree_build_doc, "build(sources_coords, sources_values, /, order=4, critical_particle_count=4, "
                                     "max_depth=20, work_order=None, alpha_centroid=0.5, n_threads=1)\n"
                                     "Build a Barnes-Hut tree from source arrays and return a new tree.\n"
                                     "\n"
                                     "Parameters\n"
                                     "----------\n"
                                     "sources_coords : (..., 3) array_like\n"
                                     "    Positions of the vortex sources.\n"
                                     "sources_values : (..., 3) array_like\n"
                                     "    Vector strengths of the sources (same batch shape as coords).\n"
                                     "order : int, default 4\n"
                                     "    Multipole order for compressed leaves.\n"
                                     "critical_particle_count : int, default 4\n"
                                     "    Base threshold for leaf subdivision.\n"
                                     "max_depth : int, default 20\n"
                                     "    Maximum tree depth.\n"
                                     "work_order : int or None, default None\n"
                                     "    Internal order for multipole shifting (\\'\\'None\\'\\' = use *order*).\n"
                                     "alpha_centroid : float, default 0.5\n"
                                     "    Centroid-based subdivision threshold (0.0 = disabled).\n"
                                     "n_threads : int, default 1\n"
                                     "    OpenMP thread count.\n"
                                     "\n"
                                     "Returns\n"
                                     "-------\n"
                                     "BarnesHutTree\n"
                                     "    Populated tree.\n"
                                     "\n"
                                     "Examples\n"
                                     "--------\n"
                                     ">>> import numpy as np\n"
                                     ">>> from pyvl.cvl import BarnesHutTree\n"
                                     ">>> rng = np.random.default_rng(1)\n"
                                     ">>> coords = rng.normal(0, 1, (200, 3))\n"
                                     ">>> vals = rng.normal(0, 1, (200, 3))\n"
                                     ">>> tree = BarnesHutTree.build(coords, vals, order=4)\n"
                                     ">>> tree.n_sources\n"
                                     "200\n");

static PyObject *pyvl_bh_tree_build(PyTypeObject *type, PyObject *const *args, const Py_ssize_t nargs,
                                    const PyObject *kwnames)
{
    const module_state_t *state = get_module_state(type);
    if (!state)
        return NULL;

    PyObject *coords_obj, *values_obj;
    Py_ssize_t order = 4, critical = 4, max_depth = 20, work_order = -1, n_threads = 1;
    double alpha_centroid = 0.5;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&coords_obj},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&values_obj},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &order, .kwname = "order", .optional = true},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &critical, .kwname = "critical_particle_count", .optional = true},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &max_depth, .kwname = "max_depth", .optional = true},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &work_order, .kwname = "work_order", .optional = true},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &alpha_centroid, .kwname = "alpha_centroid", .optional = true},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // Validate settings
    if (order < 1)
    {
        PyErr_SetString(PyExc_ValueError, "order must be >= 1.");
        return NULL;
    }
    if (critical < 1)
    {
        PyErr_SetString(PyExc_ValueError, "critical_particle_count must be >= 1.");
        return NULL;
    }
    if (max_depth < 1)
    {
        PyErr_SetString(PyExc_ValueError, "max_depth must be >= 1.");
        return NULL;
    }
    if (alpha_centroid < 0.0)
    {
        PyErr_SetString(PyExc_ValueError, "alpha_centroid must be >= 0.");
        return NULL;
    }
    if (n_threads < 1)
    {
        PyErr_SetString(PyExc_ValueError, "n_threads must be >= 1.");
        return NULL;
    }

    // Flatten source arrays
    size_t n_coords, n_vals;
    PyArrayObject *coords_arr = bh_flatten_points(coords_obj, "sources_coords", &n_coords);
    if (!coords_arr)
        return NULL;
    PyArrayObject *values_arr = bh_flatten_points(values_obj, "sources_values", &n_vals);
    if (!values_arr)
    {
        Py_DECREF(coords_arr);
        return NULL;
    }

    if (n_coords != n_vals)
    {
        PyErr_Format(PyExc_ValueError, "Number of coordinates (%zu) does not match number of values (%zu).", n_coords,
                     n_vals);
        Py_DECREF(coords_arr);
        Py_DECREF(values_arr);
        return NULL;
    }

    // Allocate and populate the Python object
    PyVL_BarnesHutObject *self = (PyVL_BarnesHutObject *)type->tp_alloc(type, 0);
    if (!self)
    {
        Py_DECREF(coords_arr);
        Py_DECREF(values_arr);
        return NULL;
    }

    self->tree = (barnes_hut_tree_t){0};
    self->n_threads = 1;
    self->sources_coords = NULL;
    self->sources_values = NULL;
    self->built = false;

    const barnes_hut_settings_t settings = {
        .order = (unsigned)order,
        .critical_particle_count = (unsigned)critical,
        .max_depth = (unsigned)max_depth,
        .work_order = resolve_py_work_order(work_order),
        .alpha_centroid = (real_t)alpha_centroid,
    };

    const bool ok =
        barnes_hut_tree_build((unsigned)n_coords, (unsigned)n_threads, (const real3_t *)PyArray_DATA(coords_arr),
                              (const real3_t *)PyArray_DATA(values_arr), &settings, &CVL_MEM_ALLOCATOR, &self->tree);
    if (!ok)
    {
        Py_DECREF(self);
        Py_DECREF(coords_arr);
        Py_DECREF(values_arr);
        PyErr_SetString(PyExc_RuntimeError, "barnes_hut_tree_build failed.");
        return NULL;
    }

    // Store source arrays (INCREF'd references).
    self->sources_coords = (PyObject *)coords_arr; // reference transferred
    self->sources_values = (PyObject *)values_arr; // reference transferred
    self->n_threads = (unsigned)n_threads;
    self->built = true;

    return (PyObject *)self;
}

/*
 * ----------------------------------------------------------------
 *  eval(self, targets, /, *, theta=0.0, n_threads=None, out=None)
 * ----------------------------------------------------------------
 */

PyDoc_STRVAR(pyvl_bh_tree_eval_doc, "eval(targets, /, *, theta=0.3, n_threads=None, out=None)\n"
                                    "Evaluate the tree at one or more target points.\n"
                                    "\n"
                                    "Parameters\n"
                                    "----------\n"
                                    "targets : (..., 3) array_like\n"
                                    "    Points at which to evaluate.  All leading dimensions are\n"
                                    "    preserved in the output.\n"
                                    "theta : float, default 0.3\n"
                                    "    Multipole acceptance criterion (MAC).\n"
                                    "    ``<= 0``  neighbour criterion: accept cell when eval point is\n"
                                    "       outside 3x3x3 neighbourhood (safest, moderate speed).\n"
                                    "    ``> 0``   opening-angle criterion: accept when\n"
                                    "       ``half_size / distance < theta``.\n"
                                    "       Recommended: 0.3 (far-field, fast), 0.01 (mid-field,\n"
                                    "       slow but accurate).\n"
                                    "n_threads : int or None, default None\n"
                                    "    OpenMP thread count.  ``None`` uses the value passed to\n"
                                    "    ``build()`` (default ``1``).\n"
                                    "out : (..., 3) ndarray, optional\n"
                                    "    Output array.  Must have the same shape as *targets*, be\n"
                                    "    writable, C-contiguous and aligned.\n"
                                    "\n"
                                    "Returns\n"
                                    "-------\n"
                                    "(..., 3) ndarray\n"
                                    "    Induced vector at each target point.\n"
                                    "\n"
                                    "See Also\n"
                                    "--------\n"
                                    "BarnesHutTree.build : Construct and populate a tree.\n"
                                    "\n"
                                    "Examples\n"
                                    "--------\n"
                                    ">>> import numpy as np\n"
                                    ">>> from pyvl.cvl import BarnesHutTree\n"
                                    ">>> rng = np.random.default_rng(42)\n"
                                    ">>> coords = rng.uniform(-1, 1, (100, 3))\n"
                                    ">>> vals = rng.uniform(-1, 1, (100, 3))\n"
                                    ">>> tree = BarnesHutTree.build(coords, vals, order=4)\n"
                                    ">>> pts = np.array([[10., 0., 0.], [0., 10., 0.]])\n"
                                    ">>> result = tree.eval(pts)\n"
                                    ">>> result.shape\n"
                                    "(2, 3)\n");

static PyObject *pyvl_bh_tree_eval(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                   const Py_ssize_t nargs, const PyObject *kwnames)
{
    PyVL_BarnesHutObject *this;
    const module_state_t *state;
    if (!ensure_bh_and_state(defining_class, self, (const PyVL_BarnesHutObject **)&this, &state))
        return NULL;

    if (!this->built)
    {
        PyErr_SetString(PyExc_RuntimeError, "BarnesHutTree has not been built. Call build() first.");
        return NULL;
    }

    PyObject *targets_obj;
    double theta = 0.3;
    Py_ssize_t n_threads = -1; // -1 means "use self->n_threads"
    PyArrayObject *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&targets_obj},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &theta, .kwname = "theta", .optional = true, .kw_only = true},
                {.type = CPYARG_TYPE_SSIZE,
                 .p_val = &n_threads,
                 .kwname = "n_threads",
                 .optional = true,
                 .kw_only = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&out,
                 .kwname = "out",
                 .optional = true,
                 .kw_only = true,
                 .type_check = &PyArray_Type},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    const unsigned nth = (n_threads < 0) ? this->n_threads : (unsigned)n_threads;

    // Validate and flatten targets
    size_t n_targets;
    PyArrayObject *targets_arr = bh_flatten_points(targets_obj, "targets", &n_targets);
    if (!targets_arr)
        return NULL;

    const int ndim = PyArray_NDIM(targets_arr);
    const npy_intp *dims = PyArray_DIMS(targets_arr);

    // Create or validate output array
    PyArrayObject *out_arr;
    if (out)
    {
        if (check_input_array(out, ndim, dims, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED, "out") < 0)
        {
            Py_DECREF(targets_arr);
            return NULL;
        }
        Py_INCREF(out);
        out_arr = out;
    }
    else
    {
        out_arr = (PyArrayObject *)PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
        if (!out_arr)
        {
            Py_DECREF(targets_arr);
            return NULL;
        }
    }

    const barnes_hut_eval_settings_t eval_cfg = {.theta = theta};

    const real3_t *targets_data = (const real3_t *)PyArray_DATA(targets_arr);
    const real3_t *src_coords = (const real3_t *)PyArray_DATA((PyArrayObject *)this->sources_coords);
    const real3_t *src_values = (const real3_t *)PyArray_DATA((PyArrayObject *)this->sources_values);
    real3_t *out_data = (real3_t *)PyArray_DATA(out_arr);

    Py_BEGIN_ALLOW_THREADS;
    barnes_hut_tree_eval_all(&this->tree, src_coords, src_values, (unsigned)n_targets, targets_data, out_data, eval_cfg,
                             nth);
    Py_END_ALLOW_THREADS;

    Py_DECREF(targets_arr);
    return (PyObject *)out_arr;
}

/*
 * ----------------------------------------------------------------
 *  Cost-model static methods
 * ----------------------------------------------------------------
 */

PyDoc_STRVAR(pyvl_bh_tree_multipole_eval_cost_doc, "multipole_eval_cost(order)\n"
                                                   "Return FLOP count for one multipole_eval call at *order*.\n"
                                                   "\n"
                                                   "Parameters\n"
                                                   "----------\n"
                                                   "order : int\n"
                                                   "    Multipole expansion order.\n"
                                                   "\n"
                                                   "Returns\n"
                                                   "-------\n"
                                                   "int\n"
                                                   "    Total floating-point operations.\n");

static PyObject *pyvl_bh_tree_multipole_eval_cost(PyObject *Py_UNUSED(self), PyObject *const *args,
                                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    Py_ssize_t order;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &order, .kwname = "order"},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (order < 0)
    {
        PyErr_SetString(PyExc_ValueError, "order must be non-negative.");
        return NULL;
    }
    return PyLong_FromSize_t(cost_model_multipole_eval((unsigned)order));
}

PyDoc_STRVAR(pyvl_bh_tree_direct_sum_cost_doc, "direct_sum_cost(n_sources)\n"
                                               "Return FLOP count for a direct sum over *n_sources*.\n"
                                               "\n"
                                               "Parameters\n"
                                               "----------\n"
                                               "n_sources : int\n"
                                               "    Number of source particles.\n"
                                               "\n"
                                               "Returns\n"
                                               "-------\n"
                                               "int\n"
                                               "    Total floating-point operations.\n");

static PyObject *pyvl_bh_tree_direct_sum_cost(PyObject *Py_UNUSED(self), PyObject *const *args, const Py_ssize_t nargs,
                                              const PyObject *kwnames)
{
    Py_ssize_t n;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n, .kwname = "n_sources"},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (n < 0)
    {
        PyErr_SetString(PyExc_ValueError, "n_sources must be non-negative.");
        return NULL;
    }
    return PyLong_FromSize_t(cost_model_direct_sum((unsigned)n));
}

PyDoc_STRVAR(pyvl_bh_tree_crossover_order_doc,
             "crossover_order(n_sources)\n"
             "Return smallest multipole order whose eval cost is below the direct sum.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "n_sources : int\n"
             "    Number of source particles.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "int\n"
             "    Minimum order that beats the direct sum, or ``2**31 - 1`` if none.\n");

static PyObject *pyvl_bh_tree_crossover_order(PyObject *Py_UNUSED(self), PyObject *const *args, const Py_ssize_t nargs,
                                              const PyObject *kwnames)
{
    Py_ssize_t n;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n, .kwname = "n_sources"},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (n < 0)
    {
        PyErr_SetString(PyExc_ValueError, "n_sources must be non-negative.");
        return NULL;
    }
    const unsigned ord = cost_model_crossover_order((unsigned)n);
    if (ord == UINT_MAX)
        return PyLong_FromLong(-1);
    return PyLong_FromUnsignedLong(ord);
}

PyDoc_STRVAR(pyvl_bh_tree_min_sources_for_order_doc,
             "min_sources_for_order(order)\n"
             "Return minimum source count where multipole at *order* beats direct sum.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "order : int\n"
             "    Multipole expansion order.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "int\n"
             "    Minimum number of sources.\n");

static PyObject *pyvl_bh_tree_min_sources_for_order(PyObject *Py_UNUSED(self), PyObject *const *args,
                                                    const Py_ssize_t nargs, const PyObject *kwnames)
{
    Py_ssize_t order;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &order, .kwname = "order"},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (order < 0)
    {
        PyErr_SetString(PyExc_ValueError, "order must be non-negative.");
        return NULL;
    }
    return PyLong_FromUnsignedLong(cost_model_min_sources_for_order((unsigned)order));
}

/*
 * ----------------------------------------------------------------
 *  Method table & type spec
 * ----------------------------------------------------------------
 */

static PyMethodDef pyvl_bh_tree_methods[] = {
    {
        .ml_name = "build",
        .ml_meth = (void *)pyvl_bh_tree_build,
        .ml_flags = METH_CLASS | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = pyvl_bh_tree_build_doc,
    },
    {
        .ml_name = "eval",
        .ml_meth = (void *)pyvl_bh_tree_eval,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = pyvl_bh_tree_eval_doc,
    },
    {
        .ml_name = "multipole_eval_cost",
        .ml_meth = (void *)pyvl_bh_tree_multipole_eval_cost,
        .ml_flags = METH_STATIC | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = pyvl_bh_tree_multipole_eval_cost_doc,
    },
    {
        .ml_name = "direct_sum_cost",
        .ml_meth = (void *)pyvl_bh_tree_direct_sum_cost,
        .ml_flags = METH_STATIC | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = pyvl_bh_tree_direct_sum_cost_doc,
    },
    {
        .ml_name = "crossover_order",
        .ml_meth = (void *)pyvl_bh_tree_crossover_order,
        .ml_flags = METH_STATIC | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = pyvl_bh_tree_crossover_order_doc,
    },
    {
        .ml_name = "min_sources_for_order",
        .ml_meth = (void *)pyvl_bh_tree_min_sources_for_order,
        .ml_flags = METH_STATIC | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = pyvl_bh_tree_min_sources_for_order_doc,
    },
    {0},
};

PyType_Spec pyvl_bh_tree_typespec = {
    .name = PYVL_CTYPE_NAME(BarnesHutTree),
    .basicsize = sizeof(PyVL_BarnesHutObject),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_str, pyvl_bh_tree_str},
            {Py_tp_repr, pyvl_bh_tree_repr},
            {Py_tp_doc, (void *)pyvl_bh_tree_type_docstring},
            {Py_tp_methods, pyvl_bh_tree_methods},
            {Py_tp_getset, pyvl_bh_tree_getset},
            {Py_tp_new, pyvl_bh_tree_new},
            {Py_tp_dealloc, pyvl_bh_tree_dealloc},
            {Py_tp_traverse, pyvl_bh_tree_traverse},
            {0},
        },
};
