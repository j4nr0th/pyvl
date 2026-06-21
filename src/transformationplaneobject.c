#include "transformationplaneobject.h"
#include "core/transformation.h"
#include <numpy/arrayobject.h>

// Must be after the array include
#include <cpyutl.h>

static bool ensure_transformation_plane_and_state(PyObject *self, PyTypeObject *defining_class,
                                                  const PyVL_TransformationPlane **p_this,
                                                  const module_state_t **p_state)
{
    const module_state_t *mod_state;
    if (!defining_class)
    {
        mod_state = get_module_state(Py_TYPE(self));
    }
    else
    {
        mod_state = PyType_GetModuleState(defining_class);
    }
    if (!mod_state)
        return false;
    *p_state = mod_state;
    if (!PyObject_TypeCheck(self, mod_state->transformation_plane_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a TransformationPlane object, got %s", Py_TYPE(self)->tp_name);
        return false;
    }
    *p_this = (const PyVL_TransformationPlane *)self;
    return true;
}

static PyArrayObject *pyvl_transformation_plane_ensure_output_array_as_input(PyArrayObject *out_array,
                                                                             const PyArrayObject *in_array)
{
    const unsigned ndim = PyArray_NDIM(in_array);
    const npy_intp *const dims = PyArray_DIMS(in_array);
    if (!out_array)
    {
        return (PyArrayObject *)PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    }

    if (check_input_array(out_array, ndim, dims, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "Output array") < 0)
        return NULL;
    Py_INCREF(out_array);
    return out_array;
}

static PyObject *pyvl_transformation_plane_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *origin_arg = NULL, *normal_arg = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", (char *[3]){"origin", "normal", NULL}, &origin_arg,
                                     &normal_arg))
        return NULL;

    PyVL_TransformationPlane *const this = (PyVL_TransformationPlane *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;

    this->origin.type = PYVL_VEC_CONSTANT;
    this->normal.type = PYVL_VEC_CONSTANT;

    if (!pyvl_vec_time_dependent_init(&this->origin, origin_arg, "origin") ||
        !pyvl_vec_time_dependent_init(&this->normal, normal_arg, "normal"))
    {
        pyvl_vec_time_dependent_clear(&this->origin);
        pyvl_vec_time_dependent_clear(&this->normal);
        Py_DECREF(this);
        return NULL;
    }

    if (this->normal.type == PYVL_VEC_CONSTANT)
    {
        // Normalize the vector if constant
        const real_t mag = real3_mag(this->normal.value.constant);
        if (mag == 0)
        {
            PyErr_SetString(PyExc_ValueError, "Normal vector cannot be zero.");
            pyvl_vec_time_dependent_clear(&this->origin);
            pyvl_vec_time_dependent_clear(&this->normal);
            Py_DECREF(this);
            return NULL;
        }
        this->normal.value.constant = real3_mul1(this->normal.value.constant, 1.0 / mag);
    }

    return (PyObject *)this;
}

static int pyvl_transformation_plane_traverse(PyObject *self, const visitproc visit, void *arg)
{
    const PyVL_TransformationPlane *const this = (PyVL_TransformationPlane *)self;
    if (this->origin.type == PYVL_VEC_CALLABLE)
        Py_VISIT(this->origin.value.callable);
    if (this->normal.type == PYVL_VEC_CALLABLE)
        Py_VISIT(this->normal.value.callable);
    return 0;
}

static void pyvl_transformation_plane_dealloc(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    PyVL_TransformationPlane *const this = (PyVL_TransformationPlane *)self;
    pyvl_vec_time_dependent_clear(&this->origin);
    pyvl_vec_time_dependent_clear(&this->normal);
    PyTypeObject *const type = Py_TYPE(this);
    type->tp_free(this);
    Py_DECREF(type);
}

static PyObject *pyvl_transformation_plane_repr(PyObject *self)
{
    const PyVL_TransformationPlane *const this = (PyVL_TransformationPlane *)self;
    const char *origin_str = this->origin.type == PYVL_VEC_CALLABLE ? "<callable>" : "(constant)";
    const char *normal_str = this->normal.type == PYVL_VEC_CALLABLE ? "<callable>" : "(constant)";
    return PyUnicode_FromFormat("TransformationPlane(origin=%s, normal=%s)", origin_str, normal_str);
}

static PyObject *pyvl_transformation_plane_origin(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_TransformationPlane *this;
    const module_state_t *state;
    if (!ensure_transformation_plane_and_state(self, defining_class, &this, &state))
        return NULL;

    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){{.type = CPYARG_TYPE_DOUBLE, .kwname = "t", .p_val = &t, .optional = true}, {0}},
            args, nargs, kwnames) < 0)
        return NULL;

    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
        return NULL;
    PyObject *const res = pyvl_vec_time_dependent_eval_py(&this->origin, time);
    Py_DECREF(time);
    return res;
}

static PyObject *pyvl_transformation_plane_normal(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_TransformationPlane *this;
    const module_state_t *state;
    if (!ensure_transformation_plane_and_state(self, defining_class, &this, &state))
        return NULL;
    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){{.type = CPYARG_TYPE_DOUBLE, .kwname = "t", .p_val = &t, .optional = true}, {0}},
            args, nargs, kwnames) < 0)
        return NULL;

    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
        return NULL;
    PyObject *const res = pyvl_vec_time_dependent_eval_py(&this->normal, time);
    Py_DECREF(time);
    if (!res)
        return NULL;

    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(res, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    Py_DECREF(res);
    if (!arr)
        return NULL;

    real3_t *const v = PyArray_DATA(arr);
    const real_t mag = real3_mag(*v);
    if (mag == 0)
    {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError, "Normal vector cannot be zero.");
        return NULL;
    }
    *v = real3_mul1(*v, 1.0 / mag);
    return (PyObject *)arr;
}

static PyObject *pyvl_transformation_plane_reflect(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                   const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_TransformationPlane *this;
    const module_state_t *state;
    if (!ensure_transformation_plane_and_state(self, defining_class, &this, &state))
        return NULL;

    PyObject *x_any;
    PyArrayObject *out = NULL;
    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){{.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&x_any},
                                  {.type = CPYARG_TYPE_DOUBLE, .kwname = "t", .p_val = &t, .optional = true},
                                  {.type = CPYARG_TYPE_PYTHON,
                                   .kwname = "out",
                                   .p_val = (void *)&out,
                                   .type_check = &PyArray_Type,
                                   .optional = true},
                                  {0}},
            args, nargs, kwnames) < 0)
        return NULL;

    PyArrayObject *const x =
        (PyArrayObject *)PyArray_FROMANY(x_any, NPY_DOUBLE, 1, INT_MAX, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!x)
        return NULL;

    const int ndim = PyArray_NDIM(x);
    if (ndim < 1 || PyArray_DIM(x, ndim - 1) != 3)
    {
        Py_DECREF(x);
        PyErr_SetString(PyExc_ValueError, "Input array must have shape (..., 3).");
        return NULL;
    }

    PyArrayObject *const out_arr = pyvl_transformation_plane_ensure_output_array_as_input(out, x);
    if (!out_arr)
    {
        Py_DECREF(x);
        return NULL;
    }

    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
    {
        Py_DECREF(x);
        Py_DECREF(out_arr);
        return NULL;
    }

    transformation_plane_t transformation_plane;
    if (!pyvl_vec_time_dependent_eval_c(&this->origin, time, &transformation_plane.origin) ||
        !pyvl_vec_time_dependent_eval_c(&this->normal, time, &transformation_plane.normal))
    {
        Py_DECREF(time);
        Py_DECREF(x);
        Py_DECREF(out_arr);
        return NULL;
    }
    Py_DECREF(time);

    size_t n = 1;
    for (int i = 0; i < ndim - 1; ++i)
        n *= PyArray_DIM(x, i);

    const real3_t *const in = PyArray_DATA(x);
    real3_t *const out_ptr = PyArray_DATA(out_arr);

    for (size_t i = 0; i < n; ++i)
    {
        out_ptr[i] = transformation_plane_transform_position(&transformation_plane, in[i]);
    }

    Py_DECREF(x);
    return (PyObject *)out_arr;
}

static PyObject *pyvl_transformation_plane_at_time(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                   const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_TransformationPlane *this;
    const module_state_t *state;
    if (!ensure_transformation_plane_and_state(self, defining_class, &this, &state))
        return NULL;

    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){{.type = CPYARG_TYPE_DOUBLE, .kwname = "t", .p_val = &t, .optional = true}, {0}},
            args, nargs, kwnames) < 0)
        return NULL;

    if (pyvl_transformation_plane_is_time_invariant(this))
    {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    real3_t origin, normal;
    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
        return NULL;

    if (!pyvl_vec_time_dependent_eval_c(&this->origin, time, &origin) ||
        !pyvl_vec_time_dependent_eval_c(&this->normal, time, &normal))
    {
        Py_DECREF(time);
        return NULL;
    }
    Py_DECREF(time);

    PyVL_TransformationPlane *const new_plane =
        (PyVL_TransformationPlane *)state->transformation_plane_type->tp_alloc(state->transformation_plane_type, 0);
    if (!new_plane)
        return NULL;

    new_plane->origin.type = PYVL_VEC_CONSTANT;
    new_plane->origin.value.constant = origin;
    new_plane->normal.type = PYVL_VEC_CONSTANT;
    new_plane->normal.value.constant = normal;

    return (PyObject *)new_plane;
}

static PyMethodDef pyvl_transformation_plane_methods[] = {
    {
        .ml_name = "origin",
        .ml_meth = (void *)pyvl_transformation_plane_origin,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "origin(t: float = 0.0) -> array\n"
                  "Get the origin of the plane at the given time.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "t : float, default: 0\n"
                  "    Time at which to evaluate the origin.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(3,) array\n"
                  "    Origin vector at the given time.\n",
    },
    {
        .ml_name = "normal",
        .ml_meth = (void *)pyvl_transformation_plane_normal,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "normal(t: float = 0.0) -> array\n"
                  "Get the normal of the plane at the given time.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "t : float, default: 0\n"
                  "    Time at which to evaluate the normal.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(3,) array\n"
                  "    Normal vector at the given time.\n",
    },
    {
        .ml_name = "reflect",
        .ml_meth = (void *)pyvl_transformation_plane_reflect,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "reflect(x: array_like, t: float = 0.0, out: out_array | None = None) -> array\n"
                  "Reflect points across the plane.\n"
                  "Parameters\n"
                  "----------\n"
                  "x : array\n"
                  "    Array of points to reflect. Must be an aligned, continuous (N, 3) array,\n"
                  "    where N is the number of points.\n"
                  "\n"
                  "t : float, default: 0\n"
                  "    Time at which to evaluate the plane's position and orientation.\n"
                  "\n"
                  "out : array, optional\n"
                  "    Array used to store the output. If not given or ``None``, a new array will\n"
                  "    be created.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "array\n"
                  "    Reflected points. If ``out`` was not ``None``, a reference to it is returned,\n"
                  "otherwise a new array is returned.\n",
    },
    {
        .ml_name = "at_time",
        .ml_meth = (void *)pyvl_transformation_plane_at_time,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "at_time(t: float = 0.0) -> TransformationPlane\nGet the plane at the given time."
                  "Get the plane at the given time.\n"
                  "\n"
                  "For planes with constant origin and normal, this will return the same plane. For\n"
                  "planes with time-dependent origin and/or normal, this will return a new plane\n"
                  "with the origin and normal evaluated at the given time.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "t : float\n"
                  "    Time at which to evaluate the plane's position and orientation.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "TransformationPlane\n"
                  "    New plane at the given time.\n",
    },
    {0},
};

PyDoc_STRVAR(pyvl_transformation_plane_type_docstring,
             "TransformationPlane(origin=(0, 0, 0), normal=(0, 0, 1))\n"
             "Type used to describe a plane used for a transformation.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "origin : VecLike3 or Callable, default: (0, 0, 0)\n"
             "    Origin of the plane. Can be a constant vector or a callable returning the origin\n"
             "    at time t.\n"
             "\n"
             "normal : VecLike3 or Callable, default: (0, 0, 1)\n"
             "    Normal of the plane. Can be a constant vector or a callable returning the normal\n"
             "    at time t. Does not need to be normalized, but it will be internally.\n");

CVL_INTERNAL
PyType_Spec pyvl_transformation_plane_typespec = {
    .name = PYVL_CTYPE_NAME(TransformationPlane),
    .basicsize = sizeof(PyVL_TransformationPlane),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_IMMUTABLETYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_doc, (void *)pyvl_transformation_plane_type_docstring},
            {Py_tp_repr, pyvl_transformation_plane_repr},
            {Py_tp_methods, pyvl_transformation_plane_methods},
            {Py_tp_new, pyvl_transformation_plane_new},
            {Py_tp_dealloc, pyvl_transformation_plane_dealloc},
            {Py_tp_traverse, pyvl_transformation_plane_traverse},
            {0, NULL},
        },
};

bool pyvl_transformation_plane_is_time_invariant(const PyVL_TransformationPlane *plane)
{
    return plane->normal.type == PYVL_VEC_CONSTANT && plane->origin.type == PYVL_VEC_CONSTANT;
}

bool pyvl_transformation_plane_ensure_time_invariant(const PyVL_TransformationPlane *plane)
{
    if (!pyvl_transformation_plane_is_time_invariant(plane))
    {
        PyErr_SetString(PyExc_ValueError, "TransformationPlane must be time-invariant.");
        return false;
    }
    return true;
}
