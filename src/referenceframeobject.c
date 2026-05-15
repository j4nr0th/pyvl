#include "referenceframeobject.h"
#include <numpy/arrayobject.h>

// Must be after the array include
#include <cpyutl.h>

/**
 * Initialize a time-dependent field with either a constant or a callable.
 *
 * @param field Field to initialize.
 * @param value Python object to initialize it from.
 * @return False on failure, with the Python exception raised.
 */
static bool init_time_dependent(pyvl_rf_time_dependent_t *const field, PyObject *value)
{
    if (value == NULL || Py_IsNone(value))
    {
        // Is it None/missing?
        field->type = PYVL_RF_CONSTANT;
        field->value.constant = (real3_t){.x = 0, .y = 0, .z = 0};
        return true;
    }

    if (PyCallable_Check(value))
    {
        // Do we have a callable?
        field->type = PYVL_RF_CALLABLE;
        field->value.callable = value;
        Py_INCREF(value);
        return true;
    }

    // Well, it should be a vector of 3 entries
    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FromAny(value, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!arr)
    {
        PyErr_SetString(PyExc_TypeError, "Field must be a callable or a sequence of 3 numbers.");
        return false;
    }
    const npy_intp n = PyArray_SIZE(arr);
    if (n != 3)
    {
        Py_DECREF(arr);
        PyErr_Format(PyExc_ValueError, "Expected 3 elements, got %d.", (int)n);
        return false;
    }
    const double *const p = PyArray_DATA(arr);
    // Extract the constant value
    field->type = PYVL_RF_CONSTANT;
    field->value.constant = (real3_t){.x = p[0], .y = p[1], .z = p[2]};
    Py_DECREF(arr);
    return true;
}

/**
 * Evaluate the field callable at a time and check the return has the correct type.
 *
 * @param field Callable field to call.
 * @param time_arg Time to evaluate at. Should be PyFloat.
 * @return Array of 3 doubles with the value of the field at the specified time.
 */
static PyArrayObject *evaluate_time_dependent_callable(const pyvl_rf_time_dependent_t *field, PyObject *const time_arg)
{
    PyObject *const res = PyObject_Vectorcall(field->value.callable, (PyObject *const[1]){time_arg}, 1, NULL);
    if (!res)
        return NULL;

    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(res, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    Py_DECREF(res);

    if (!arr)
    {
        PyErr_SetString(PyExc_TypeError, "Callable must return a sequence of 3 numbers.");
        return NULL;
    }
    const npy_intp n = PyArray_SIZE(arr);
    if (n != 3)
    {
        Py_DECREF(arr);
        PyErr_Format(PyExc_ValueError, "Expected 3 elements, got %d.", (int)n);
        return NULL;
    }
    return arr;
}

static bool evaluate_time_dependent_c(const pyvl_rf_time_dependent_t *field, PyObject *t, real3_t *val)
{
    if (field->type == PYVL_RF_CONSTANT)
    {
        *val = field->value.constant;
        return true;
    }

    PyArrayObject *const arr = evaluate_time_dependent_callable(field, t);
    if (!arr)
        return false;

    *val = *(real3_t *)PyArray_DATA(arr);
    Py_DECREF(arr);

    return true;
}

static PyObject *evaluate_time_dependent_python(const pyvl_rf_time_dependent_t *field, PyObject *t)
{
    if (field->type == PYVL_RF_CALLABLE)
    {
        return (PyObject *)evaluate_time_dependent_callable(field, t);
    }
    static const npy_intp size = 3;
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if (!out)
        return NULL;
    real3_t *const p_out = PyArray_DATA(out);
    *p_out = field->value.constant;
    return (PyObject *)out;
}

static void clear_time_dependent(pyvl_rf_time_dependent_t *field)
{
    if (field->type == PYVL_RF_CALLABLE)
    {
        // Decref
        Py_DECREF(field->value.callable);
    }
    // Clear it and set it to zero
    field->type = PYVL_RF_CONSTANT;
    field->value.constant = (real3_t){0};
}

static PyObject *pyvl_reference_frame_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    const module_state_t *const state = get_module_state(type);
    if (!state)
        return NULL;

    PyObject *offset_arg = NULL, *theta_arg = NULL, *velocity_arg = NULL, *rotation_arg = NULL;
    PyObject *parent_arg = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOOO",
                                     (char *[6]){"offset", "theta", "velocity", "rotation", "parent", NULL},
                                     &offset_arg, &theta_arg, &velocity_arg, &rotation_arg, &parent_arg))
    {
        return NULL;
    }

    PyVL_ReferenceFrame *parent = NULL;
    if (parent_arg != NULL && !Py_IsNone(parent_arg))
    {
        if (!PyObject_TypeCheck(parent_arg, state->rf_type))
        {
            PyErr_Format(PyExc_TypeError, "Argument \"parent\" must be a ReferenceFrame object, but it was %R",
                         Py_TYPE(parent_arg));
            return NULL;
        }
        parent = (PyVL_ReferenceFrame *)parent_arg;
    }

    PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;

    // Set this so that the clear function is guaranteed to work correctly.
    this->position.type = PYVL_RF_CONSTANT;
    this->velocity.type = PYVL_RF_CONSTANT;
    this->orientation.type = PYVL_RF_CONSTANT;
    this->rotation.type = PYVL_RF_CONSTANT;

    if (!init_time_dependent(&this->position, offset_arg) || !init_time_dependent(&this->orientation, theta_arg) ||
        !init_time_dependent(&this->velocity, velocity_arg) || !init_time_dependent(&this->rotation, rotation_arg))
    {
        clear_time_dependent(&this->position);
        clear_time_dependent(&this->orientation);
        clear_time_dependent(&this->velocity);
        clear_time_dependent(&this->rotation);
        Py_DECREF(this);
        return NULL;
    }
    this->parent = parent;
    Py_XINCREF(parent);

    return (PyObject *)this;
}

static int pyvl_reference_frame_traverse(PyObject *self, const visitproc visit, void *arg)
{
    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    if (this->position.type == PYVL_RF_CALLABLE)
        Py_VISIT(this->position.value.callable);
    if (this->velocity.type == PYVL_RF_CALLABLE)
        Py_VISIT(this->velocity.value.callable);
    if (this->orientation.type == PYVL_RF_CALLABLE)
        Py_VISIT(this->orientation.value.callable);
    if (this->rotation.type == PYVL_RF_CALLABLE)
        Py_VISIT(this->rotation.value.callable);
    Py_VISIT(this->parent);
    return 0;
}

static void pyvl_reference_frame_dealloc(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    clear_time_dependent(&this->position);
    clear_time_dependent(&this->velocity);
    clear_time_dependent(&this->orientation);
    clear_time_dependent(&this->rotation);
    Py_XDECREF(this->parent);
    this->parent = NULL;
    PyTypeObject *type = Py_TYPE(this);
    type->tp_free(this);
    Py_DECREF(type);
}

static PyObject *pyvl_reference_frame_repr(PyObject *self)
{
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    const char *pos_str = this->position.type == PYVL_RF_CALLABLE ? "<callable>" : "(constant)";
    const char *ori_str = this->orientation.type == PYVL_RF_CALLABLE ? "<callable>" : "(constant)";
    PyObject *out;
    if (this->parent)
    {
        out = PyUnicode_FromFormat("ReferenceFrame(position=%s, orientation=%s, parent=%R)", pos_str, ori_str,
                                   (PyObject *)this->parent);
    }
    else
    {
        out = PyUnicode_FromFormat("ReferenceFrame(position=%s, orientation=%s)", pos_str, ori_str);
    }
    return out;
}

static PyObject *pyvl_reference_frame_get_parent(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    if (!this->parent)
    {
        Py_RETURN_NONE;
    }
    Py_INCREF(this->parent);
    return (PyObject *)this->parent;
}

static PyObject *pyvl_reference_frame_get_parents(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    unsigned parent_cnt = 0;
    for (const PyVL_ReferenceFrame *p = this->parent; p; p = p->parent)
        parent_cnt += 1;
    PyObject *out = PyTuple_New(parent_cnt);
    unsigned i = 0;
    for (PyVL_ReferenceFrame *p = this->parent; p; p = p->parent)
    {
        Py_INCREF(p);
        PyTuple_SET_ITEM(out, i, (PyObject *)p);
        i += 1;
    }
    return out;
}

static PyObject *pyvl_reference_frame_rich_compare(PyObject *self, PyObject *other, const int op)
{
    const module_state_t *const state = get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;

    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    if (Py_TYPE(other) != state->rf_type)
    {
        Py_RETURN_FALSE;
    }
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    const PyVL_ReferenceFrame *that = (PyVL_ReferenceFrame *)other;

    bool result = true;
    while (this && that)
    {
        if (this->position.type != that->position.type)
        {
            result = false;
            break;
        }
        if (this->position.type == PYVL_RF_CONSTANT)
        {
            const real3_t dr = real3_sub(this->position.value.constant, that->position.value.constant);
            if (real3_dot(dr, dr) > 1e-20)
            {
                result = false;
                break;
            }
        }
        if (this->orientation.type != that->orientation.type)
        {
            result = false;
            break;
        }
        if (this->orientation.type == PYVL_RF_CONSTANT)
        {
            const real3_t dr = real3_sub(this->orientation.value.constant, that->orientation.value.constant);
            if (real3_dot(dr, dr) > 1e-20)
            {
                result = false;
                break;
            }
        }
        this = this->parent;
        that = that->parent;
    }
    if (this || that)
    {
        result = false;
    }

    result = (op == Py_EQ ? (int)result : !result) != 0;
    if (!result)
    {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyGetSetDef pyvl_reference_frame_getset[] = {
    {
        .name = "parent",
        .get = pyvl_reference_frame_get_parent,
        .doc = "ReferenceFrame | None : Return what reference frame the current one is defined relative to.\n",
    },
    {
        .name = "parents",
        .get = pyvl_reference_frame_get_parents,
        .doc = "tuple[ReferenceFrame, ...] : Tuple of all parents of this reference frame.\n",
    },
    {0},
};

static bool get_transformation_at_time(const PyVL_ReferenceFrame *this, PyObject *t, real3x3_t *p_mat, real3_t *p_off)
{
    real3_t angles;
    if (!evaluate_time_dependent_c(&this->orientation, t, &angles))
        return false;

    *p_mat = real3x3_from_angles(angles);

    if (!evaluate_time_dependent_c(&this->position, t, p_off))
        return false;

    return true;
}

static bool get_global_transformation(const PyVL_ReferenceFrame *this, PyObject *t, real3x3_t *p_mat, real3_t *p_off)
{
    if (!get_transformation_at_time(this, t, p_mat, p_off))
        return false;

    for (const PyVL_ReferenceFrame *p = this; p->parent; p = p->parent)
    {
        real3x3_t p_mat_parent;
        real3_t p_off_parent;
        if (!get_transformation_at_time(p->parent, t, &p_mat_parent, &p_off_parent))
            return false;
        merge_transformations(p_mat_parent, p_off_parent, *p_mat, *p_off, p_mat, p_off);
    }

    return true;
}

static bool get_transformation_at_time_reverse(const PyVL_ReferenceFrame *this, PyObject *t, real3x3_t *p_mat,
                                               real3_t *p_off)
{
    real3_t angles;
    if (!evaluate_time_dependent_c(&this->orientation, t, &angles))
        return false;

    *p_mat = real3x3_inverse_from_angles(angles);

    if (!evaluate_time_dependent_c(&this->position, t, p_off))
        return false;

    return true;
}

static bool get_global_transformation_reverse(const PyVL_ReferenceFrame *this, PyObject *t, real3x3_t *p_mat,
                                              real3_t *p_off)
{
    if (!get_transformation_at_time_reverse(this, t, p_mat, p_off))
        return false;

    for (const PyVL_ReferenceFrame *rf = this; rf->parent; rf = rf->parent)
    {
        real3x3_t p_mat_parent;
        real3_t p_off_parent;
        if (!get_transformation_at_time_reverse(rf->parent, t, &p_mat_parent, &p_off_parent))
            return false;

        merge_transformations_reverse(p_mat_parent, p_off_parent, *p_mat, *p_off, p_mat, p_off);
    }

    return true;
}

static bool ensure_rf_and_state(PyObject *self, PyTypeObject *defining_class, const PyVL_ReferenceFrame **p_this,
                                const module_state_t **p_state)
{
    const module_state_t *state = NULL;
    if (defining_class)
    {
        state = PyType_GetModuleState(defining_class);
        if (!state)
            return false;
    }
    else
    {
        PyObject *const mod = PyType_GetModuleByDef(Py_TYPE(self), &cvl_module);
        if (!mod)
            return false;
        state = PyModule_GetState(mod);
        if (!state)
            return false;
    }

    if (!PyObject_TypeCheck(self, state->rf_type))
    {
        PyErr_Format(PyExc_TypeError, "Self was not \"%s\" but was instead \"%s\".", state->rf_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return false;
    }

    *p_this = (PyVL_ReferenceFrame *)self;
    *p_state = state;

    return true;
}

static PyObject *pyvl_reference_frame_offset_at(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .optional = 1},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
        return NULL;
    PyObject *const res = evaluate_time_dependent_python(&this->position, time);
    Py_DECREF(time);
    return res;
}

static PyObject *pyvl_reference_frame_velocity_at(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .optional = 1},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
        return NULL;
    PyObject *const res = evaluate_time_dependent_python(&this->velocity, time);
    Py_DECREF(time);
    return res;
}

static PyObject *pyvl_reference_frame_angles_at(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .optional = 1},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
        return NULL;
    PyObject *const res = evaluate_time_dependent_python(&this->orientation, time);
    Py_DECREF(time);
    return res;
}

static PyObject *pyvl_reference_frame_rotation_at(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .optional = 1},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
        return NULL;
    PyObject *const res = evaluate_time_dependent_python(&this->rotation, time);
    Py_DECREF(time);
    return res;
}

static PyObject *pyvl_reference_frame_rotation_matrix_at(PyObject *self, PyTypeObject *defining_class,
                                                         PyObject *const *args, const Py_ssize_t nargs,
                                                         const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .optional = 1},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
        return NULL;
    real3_t angles;
    const bool r = evaluate_time_dependent_c(&this->orientation, time, &angles);
    Py_DECREF(time);
    if (!r)
        return NULL;

    static const npy_intp dims[2] = {3, 3};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
        return NULL;
    real3x3_t *const p_out = PyArray_DATA(out);

    *p_out = real3x3_from_angles(angles);
    return (PyObject *)out;
}

static PyObject *pyvl_reference_frame_rotation_matrix_inverse_at(PyObject *self, PyTypeObject *defining_class,
                                                                 PyObject *const *args, const Py_ssize_t nargs,
                                                                 const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .optional = 1},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
        return NULL;
    real3_t angles;
    const bool r = evaluate_time_dependent_c(&this->orientation, time, &angles);
    Py_DECREF(time);
    if (!r)
        return NULL;

    static const npy_intp dims[2] = {3, 3};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
        return NULL;
    real3x3_t *const p_out = PyArray_DATA(out);

    *p_out = real3x3_inverse_from_angles(angles);
    return (PyObject *)out;
}

static bool prepare_for_transformation_with_time(PyObject *const *args, const Py_ssize_t nargs, const PyObject *kwnames,
                                                 PyArrayObject **p_in, PyArrayObject **p_out, npy_intp *p_dim,
                                                 const npy_intp **p_dims, PyObject **p_time)
{
    PyObject *in_any;
    PyArrayObject *out_array = NULL;
    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&in_any, .kwname = "x"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .kwname = "time", .optional = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&out_array,
                 .kwname = "out",
                 .type_check = &PyArray_Type,
                 .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return false;

    PyArrayObject *const in_array =
        (PyArrayObject *)PyArray_FROMANY(in_any, NPY_DOUBLE, 1, INT_MAX, NPY_ARRAY_C_CONTIGUOUS);
    if (!in_array)
    {
        return false;
    }
    const npy_intp dim_in = PyArray_NDIM(in_array);
    const npy_intp *dims_in = PyArray_DIMS(in_array);
    if (dims_in[dim_in - 1] != 3)
    {
        PyErr_Format(PyExc_ValueError,
                     "Input array does not have the last axis with 3 dimensions "
                     "(shape is (..., %u) instead of (..., 3)).",
                     (unsigned)dims_in[dim_in - 1]);
        Py_DECREF(in_array);
        return false;
    }
    if (out_array)
    {
        if (check_input_array(out_array, dim_in, dims_in, NPY_DOUBLE,
                              NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "out") < 0)
        {
            Py_DECREF(in_array);
            return false;
        }
        Py_INCREF(out_array);
    }
    else
    {
        out_array = (PyArrayObject *)PyArray_NewLikeArray(in_array, NPY_CORDER, NULL, 0);
        if (!out_array)
        {
            Py_DECREF(in_array);
            return false;
        }
    }
    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
    {
        Py_DECREF(in_array);
        Py_DECREF(out_array);
        return false;
    }

    *p_in = in_array;
    *p_out = out_array;
    *p_dim = dim_in;
    *p_dims = dims_in;
    *p_time = time;
    return true;
}

static PyObject *pyvl_reference_frame_from_parent_with_offset(PyObject *self, PyTypeObject *defining_class,
                                                              PyObject *const *args, const Py_ssize_t nargs,
                                                              PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    PyObject *t;
    if (!prepare_for_transformation_with_time(args, nargs, kwnames, &in_array, &out_array, &dim_in, &dims_in, &t))
    {
        return NULL;
    }

    real3x3_t mat;
    real3_t off;
    const bool r = get_transformation_at_time(this, t, &mat, &off);
    Py_DECREF(t);
    if (!r)
    {
        Py_DECREF(in_array);
        Py_DECREF(out_array);
        return NULL;
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3_add(real3x3_vecmul(mat, p_in[i]), off);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_from_parent_without_offset(PyObject *self, PyTypeObject *defining_class,
                                                                 PyObject *const *args, const Py_ssize_t nargs,
                                                                 PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    PyObject *t;
    if (!prepare_for_transformation_with_time(args, nargs, kwnames, &in_array, &out_array, &dim_in, &dims_in, &t))
    {
        return NULL;
    }

    real3x3_t mat;
    real3_t off;
    const bool r = get_transformation_at_time(this, t, &mat, &off);
    Py_DECREF(t);
    if (!r)
    {
        Py_DECREF(in_array);
        Py_DECREF(out_array);
        return NULL;
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3x3_vecmul(mat, p_in[i]);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_to_parent_with_offset(PyObject *self, PyTypeObject *defining_class,
                                                            PyObject *const *args, const Py_ssize_t nargs,
                                                            PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    PyObject *t;
    if (!prepare_for_transformation_with_time(args, nargs, kwnames, &in_array, &out_array, &dim_in, &dims_in, &t))
    {
        return NULL;
    }

    real3x3_t mat;
    real3_t off;
    const bool r = get_transformation_at_time_reverse(this, t, &mat, &off);
    Py_DECREF(t);
    if (!r)
    {
        Py_DECREF(in_array);
        Py_DECREF(out_array);
        return NULL;
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3x3_vecmul(mat, real3_sub(p_in[i], off));
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_to_parent_without_offset(PyObject *self, PyTypeObject *defining_class,
                                                               PyObject *const *args, const Py_ssize_t nargs,
                                                               PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    PyObject *t;
    if (!prepare_for_transformation_with_time(args, nargs, kwnames, &in_array, &out_array, &dim_in, &dims_in, &t))
    {
        return NULL;
    }

    real3x3_t mat;
    real3_t off;
    const bool r = get_transformation_at_time_reverse(this, t, &mat, &off);
    Py_DECREF(t);
    if (!r)
    {
        Py_DECREF(in_array);
        Py_DECREF(out_array);
        return NULL;
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3x3_vecmul(mat, p_in[i]);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_from_global_with_offset(PyObject *self, PyTypeObject *defining_class,
                                                              PyObject *const *args, const Py_ssize_t nargs,
                                                              PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    PyObject *t;
    if (!prepare_for_transformation_with_time(args, nargs, kwnames, &in_array, &out_array, &dim_in, &dims_in, &t))
    {
        return NULL;
    }

    real3x3_t mat;
    real3_t off;
    const bool r = get_global_transformation(this, t, &mat, &off);
    Py_DECREF(t);
    if (!r)
    {
        Py_DECREF(in_array);
        Py_DECREF(out_array);
        return NULL;
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3_add(real3x3_vecmul(mat, p_in[i]), off);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_from_global_without_offset(PyObject *self, PyTypeObject *defining_class,
                                                                 PyObject *const *args, const Py_ssize_t nargs,
                                                                 PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    PyObject *t;
    if (!prepare_for_transformation_with_time(args, nargs, kwnames, &in_array, &out_array, &dim_in, &dims_in, &t))
    {
        return NULL;
    }

    real3x3_t mat;
    real3_t off;
    const bool r = get_global_transformation(this, t, &mat, &off);
    Py_DECREF(t);
    if (!r)
    {
        Py_DECREF(in_array);
        Py_DECREF(out_array);
        return NULL;
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3x3_vecmul(mat, p_in[i]);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_to_global_with_offset(PyObject *self, PyTypeObject *defining_class,
                                                            PyObject *const *args, const Py_ssize_t nargs,
                                                            PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    PyObject *t;
    if (!prepare_for_transformation_with_time(args, nargs, kwnames, &in_array, &out_array, &dim_in, &dims_in, &t))
    {
        return NULL;
    }

    real3x3_t mat;
    real3_t off;
    const bool r = get_global_transformation_reverse(this, t, &mat, &off);
    Py_DECREF(t);
    if (!r)
    {
        Py_DECREF(in_array);
        Py_DECREF(out_array);
        return NULL;
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3_sub(real3x3_vecmul(mat, p_in[i]), off);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_to_global_without_offset(PyObject *self, PyTypeObject *defining_class,
                                                               PyObject *const *args, const Py_ssize_t nargs,
                                                               PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    PyArrayObject *in_array, *out_array;
    npy_intp dim_in;
    const npy_intp *dims_in;
    PyObject *t;
    if (!prepare_for_transformation_with_time(args, nargs, kwnames, &in_array, &out_array, &dim_in, &dims_in, &t))
    {
        return NULL;
    }

    real3x3_t mat;
    real3_t off;
    const bool r = get_global_transformation_reverse(this, t, &mat, &off);
    Py_DECREF(t);
    if (!r)
    {
        Py_DECREF(in_array);
        Py_DECREF(out_array);
        return NULL;
    }

    _Static_assert(sizeof(real_t) == sizeof(npy_float64), "Binary compatibility must be ensured.");
    size_t n_entries = 1;
    for (unsigned i = 0; i < (unsigned)dim_in - 1; ++i)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    for (size_t i = 0; i < n_entries; ++i)
    {
        p_out[i] = real3x3_vecmul(mat, p_in[i]);
    }

    Py_DECREF(in_array);
    return (PyObject *)out_array;
}

static PyObject *pyvl_reference_frame_rotate_x(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                               const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    if (this->orientation.type == PYVL_RF_CALLABLE)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot rotate a ReferenceFrame with time-varying orientation.");
        return NULL;
    }
    double theta;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &theta},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyVL_ReferenceFrame *const new = (PyVL_ReferenceFrame *)state->rf_type->tp_alloc(state->rf_type, 0);
    if (!new)
        return NULL;
    new->position = this->position;
    new->velocity = this->velocity;
    new->orientation = this->orientation;
    new->rotation = this->rotation;
    Py_XINCREF(this->parent);
    new->parent = this->parent;
    new->orientation.value.constant.x = clamp_angle_to_range(new->orientation.value.constant.x + theta);
    return (PyObject *)new;
}

static PyObject *pyvl_reference_frame_rotate_y(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                               const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    if (this->orientation.type == PYVL_RF_CALLABLE)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot rotate a ReferenceFrame with time-varying orientation.");
        return NULL;
    }
    double theta;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &theta},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyVL_ReferenceFrame *const new = (PyVL_ReferenceFrame *)state->rf_type->tp_alloc(state->rf_type, 0);
    if (!new)
        return NULL;
    new->position = this->position;
    new->velocity = this->velocity;
    new->orientation = this->orientation;
    new->rotation = this->rotation;
    Py_XINCREF(this->parent);
    new->parent = this->parent;
    new->orientation.value.constant.y = clamp_angle_to_range(new->orientation.value.constant.y + theta);
    return (PyObject *)new;
}

static PyObject *pyvl_reference_frame_rotate_z(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                               const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    if (this->orientation.type == PYVL_RF_CALLABLE)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot rotate a ReferenceFrame with time-varying orientation.");
        return NULL;
    }
    double theta;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &theta},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyVL_ReferenceFrame *const new = (PyVL_ReferenceFrame *)state->rf_type->tp_alloc(state->rf_type, 0);
    if (!new)
        return NULL;
    new->position = this->position;
    new->velocity = this->velocity;
    new->orientation = this->orientation;
    new->rotation = this->rotation;
    Py_XINCREF(this->parent);
    new->parent = this->parent;
    new->orientation.value.constant.z = clamp_angle_to_range(new->orientation.value.constant.z + theta);
    return (PyObject *)new;
}

static PyObject *pyvl_reference_frame_with_offset(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    if (this->position.type == PYVL_RF_CALLABLE)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot change offset of a ReferenceFrame with time-varying position.");
        return NULL;
    }
    PyObject *in_any;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&in_any},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    PyArrayObject *const off_array = (PyArrayObject *)PyArray_FromAny(in_any, PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
                                                                      NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!off_array)
        return NULL;
    const npy_intp *p_n = PyArray_DIMS(off_array);
    if (*p_n != 3)
    {
        PyErr_Format(PyExc_ValueError, "Input array must have 3 element, instead %u were given.", (unsigned)*p_n);
        Py_DECREF(off_array);
        return NULL;
    }
    const npy_float64 *p_in = PyArray_DATA(off_array);
    Py_DECREF(off_array);

    PyVL_ReferenceFrame *const new = (PyVL_ReferenceFrame *)state->rf_type->tp_alloc(state->rf_type, 0);
    if (!new)
        return NULL;
    new->position = this->position;
    new->velocity = this->velocity;
    new->orientation = this->orientation;
    new->rotation = this->rotation;
    new->position.type = PYVL_RF_CONSTANT;
    new->position.value.constant = (real3_t){.x = p_in[0], .y = p_in[1], .z = p_in[2]};
    Py_XINCREF(this->parent);
    new->parent = this->parent;
    return (PyObject *)new;
}

static PyObject *pyvl_matrix_to_angles(PyObject *Py_UNUSED(module), PyObject *arg)
{
    PyArrayObject *const array = (PyArrayObject *)PyArray_FROMANY(arg, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!array)
        return NULL;
    const npy_intp *const dims = PyArray_DIMS(array);
    if (dims[0] != 3 || dims[1] != 3)
    {
        PyErr_Format(PyExc_ValueError, "Array was not a (3, 3) array, instead it was (%u, %u).", (unsigned)dims[0],
                     (unsigned)dims[1]);
        return NULL;
    }
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, (const npy_intp[1]){3}, NPY_DOUBLE);
    if (!out)
    {
        Py_DECREF(array);
        return NULL;
    }
    const double *restrict mat = PyArray_DATA(array);
    double *const restrict angles = PyArray_DATA(out);

    const real3x3_t matrix = {
        .m00 = mat[0],
        .m01 = mat[1],
        .m02 = mat[2],
        .m10 = mat[3],
        .m11 = mat[4],
        .m12 = mat[5],
        .m20 = mat[6],
        .m21 = mat[7],
        .m22 = mat[8],
    };
    const real3_t a = angles_from_real3x3(matrix);
    angles[0] = a.x;
    angles[1] = a.y;
    angles[2] = a.z;

    Py_DECREF(array);
    return (PyObject *)out;
}

static PyObject *pyvl_reference_frame_save(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                           const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *const state = PyType_GetModuleState(defining_class);
    if (!state)
        return NULL;

    PyObject *arg;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&arg},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (!PyMapping_Check(arg))
    {
        PyErr_Format(PyExc_TypeError, "The input parameter is not a mapping.");
        return NULL;
    }
    const PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)self;
    PyObject *const t = PyFloat_FromDouble(0);
    if (!t)
        return NULL;
    PyObject *const pos_array = evaluate_time_dependent_python(&this->position, t);
    PyObject *const ori_array = evaluate_time_dependent_python(&this->orientation, t);
    Py_DECREF(t);
    if (!pos_array || !ori_array)
    {
        Py_XDECREF(pos_array);
        Py_XDECREF(ori_array);
        return NULL;
    }

    const int res1 = PyMapping_SetItemString(arg, "offset", pos_array);
    Py_DECREF(pos_array);
    const int res2 = PyMapping_SetItemString(arg, "angles", ori_array);
    Py_DECREF(ori_array);
    if (res1 < 0 || res2 < 0)
        return NULL;
    PyObject *type_name = PyUnicode_FromString(state->rf_type->tp_name);
    if (!type_name)
        return NULL;
    const int res3 = PyMapping_SetItemString(arg, "type", type_name);
    Py_DECREF(type_name);
    if (res3 < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject *pyvl_reference_frame_load(PyTypeObject *type, PyObject *const *args, const Py_ssize_t nargs,
                                           const PyObject *kwnames)
{
    const module_state_t *const state = get_module_state(type);
    if (!state)
        return NULL;

    PyObject *group, *parent = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&group, .kwname = "group"},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&parent,
                 .kwname = "parent",
                 .type_check = state->rf_type,
                 .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (!PyMapping_Check(group))
    {
        PyErr_Format(PyExc_TypeError, "The input parameter is not a mapping.");
        return NULL;
    }

    PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;

    PyObject *const off_val = PyMapping_GetItemString(group, "offset");
    PyObject *const rot_val = PyMapping_GetItemString(group, "angles");
    if (!off_val || !rot_val)
    {
        Py_XDECREF(off_val);
        Py_XDECREF(rot_val);
        Py_DECREF(this);
        return NULL;
    }

    PyArrayObject *const off_array = (PyArrayObject *)PyArray_FromAny(off_val, PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
                                                                      NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    Py_DECREF(off_val);
    PyArrayObject *const rot_array = (PyArrayObject *)PyArray_FromAny(rot_val, PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
                                                                      NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    Py_DECREF(rot_val);
    if (!off_array || !rot_array)
    {
        Py_XDECREF(off_array);
        Py_XDECREF(rot_array);
        Py_DECREF(this);
        return NULL;
    }

    if (PyArray_SIZE(off_array) != 3)
    {
        PyErr_Format(PyExc_ValueError, "Offset array did not have 3 elements, but had %u instead.",
                     (unsigned)PyArray_SIZE(off_array));
        Py_DECREF(off_array);
        Py_DECREF(rot_array);
        Py_DECREF(this);
        return NULL;
    }

    if (PyArray_SIZE(rot_array) != 3)
    {
        PyErr_Format(PyExc_ValueError, "Angle array did not have 3 elements, but had %u instead.",
                     (unsigned)PyArray_SIZE(rot_array));
        Py_DECREF(off_array);
        Py_DECREF(rot_array);
        Py_DECREF(this);
        return NULL;
    }

    const npy_float64 *const offset_ptr = PyArray_DATA(off_array);
    const npy_float64 *const angles_ptr = PyArray_DATA(rot_array);

    this->position.type = PYVL_RF_CONSTANT;
    this->position.value.constant = (real3_t){.x = offset_ptr[0], .y = offset_ptr[1], .z = offset_ptr[2]};
    this->orientation.type = PYVL_RF_CONSTANT;
    this->orientation.value.constant = (real3_t){.x = angles_ptr[0], .y = angles_ptr[1], .z = angles_ptr[2]};
    this->velocity.type = PYVL_RF_CONSTANT;
    this->velocity.value.constant = (real3_t){.x = 0, .y = 0, .z = 0};
    this->rotation.type = PYVL_RF_CONSTANT;
    this->rotation.value.constant = (real3_t){.x = 0, .y = 0, .z = 0};

    Py_DECREF(off_array);
    Py_DECREF(rot_array);
    this->parent = (PyVL_ReferenceFrame *)parent;
    Py_XINCREF(parent);

    return (PyObject *)this;
}

static PyMethodDef pyvl_reference_frame_methods[] = {
    {
        .ml_name = "offset_at",
        .ml_meth = (void *)pyvl_reference_frame_offset_at,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "offset_at(t: float = 0.0, /) -> array\n"
                  "Get the position of the reference frame at the given time.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "t : float, default: 0.0\n"
                  "    Time at which to evaluate the position.\n"
                  "Returns\n"
                  "-------\n"
                  "(3,) array\n"
                  "    Position vector at the given time.",
    },
    {
        .ml_name = "velocity_at",
        .ml_meth = (void *)pyvl_reference_frame_velocity_at,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "velocity_at(t: float = 0.0, /) -> array\n"
                  "Get the linear velocity of the reference frame at the given time.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "t : float, default: 0.0\n"
                  "    Time at which to evaluate the velocity.\n"
                  "Returns\n"
                  "-------\n"
                  "(3,) array\n"
                  "    Velocity vector at the given time.",
    },
    {
        .ml_name = "angles_at",
        .ml_meth = (void *)pyvl_reference_frame_angles_at,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "angles_at(t: float = 0.0, /) -> array\n"
                  "Get the orientation (Euler angles) of the reference frame at the given time.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "t : float, default: 0.0\n"
                  "    Time at which to evaluate the orientation.\n"
                  "Returns\n"
                  "-------\n"
                  "(3,) array\n"
                  "    Euler angles at the given time.",
    },
    {
        .ml_name = "rotation_at",
        .ml_meth = (void *)pyvl_reference_frame_rotation_at,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "rotation_at(t: float = 0.0, /) -> array\n"
                  "Get the angular velocity (rotation rate) of the reference frame at the given time.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "t : float, default: 0.0\n"
                  "    Time at which to evaluate the rotation.\n"
                  "Returns\n"
                  "-------\n"
                  "(3,) array\n"
                  "    Angular velocity vector at the given time.",
    },
    {
        .ml_name = "rotation_matrix_at",
        .ml_meth = (void *)pyvl_reference_frame_rotation_matrix_at,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "rotation_matrix_at(t: float = 0.0, /) -> array\n"
                  "Get the rotation matrix of the reference frame at the given time.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "t : float, default: 0.0\n"
                  "    Time at which to evaluate the rotation matrix.\n"
                  "Returns\n"
                  "-------\n"
                  "(3, 3) array\n"
                  "    Rotation matrix at the given time.",
    },
    {
        .ml_name = "rotation_matrix_inverse_at",
        .ml_meth = (void *)pyvl_reference_frame_rotation_matrix_inverse_at,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "rotation_matrix_inverse_at(t: float = 0.0, /) -> array\n"
                  "Get the inverse rotation matrix of the reference frame at the given time.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "t : float, default: 0.0\n"
                  "    Time at which to evaluate the inverse rotation matrix.\n"
                  "Returns\n"
                  "-------\n"
                  "(3, 3) array\n"
                  "    Inverse rotation matrix at the given time.",
    },
    {
        .ml_name = "from_parent_with_offset",
        .ml_meth = (void *)pyvl_reference_frame_from_parent_with_offset,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_parent_with_offset(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map position vector from parent reference frame to the child reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in parent reference frame.\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the child reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "from_parent_without_offset",
        .ml_meth = (void *)pyvl_reference_frame_from_parent_without_offset,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_parent_without_offset(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map direction vector from parent reference frame to the child reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in parent reference frame.\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Direction vectors mapped to the child reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "to_parent_with_offset",
        .ml_meth = (void *)pyvl_reference_frame_to_parent_with_offset,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "to_parent_with_offset(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map position vector from child reference frame to the parent reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in child reference frame.\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the parent reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "to_parent_without_offset",
        .ml_meth = (void *)pyvl_reference_frame_to_parent_without_offset,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "to_parent_without_offset(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map direction vector from child reference frame to the parent reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in child reference frame.\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Direction vectors mapped to the parent reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "from_global_with_offset",
        .ml_meth = (void *)pyvl_reference_frame_from_global_with_offset,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_global_with_offset(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map position vector from global reference frame to the child reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in global reference frame.\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the child reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "from_global_without_offset",
        .ml_meth = (void *)pyvl_reference_frame_from_global_without_offset,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_global_without_offset(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map direction vector from global reference frame to the child reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in global reference frame.\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Direction vectors mapped to the child reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "to_global_with_offset",
        .ml_meth = (void *)pyvl_reference_frame_to_global_with_offset,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "to_global_with_offset(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map position vector from child reference frame to the global reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in child reference frame.\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the global reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "to_global_without_offset",
        .ml_meth = (void *)pyvl_reference_frame_to_global_without_offset,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "to_global_without_offset(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map direction vector from child reference frame to the global reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in child reference frame.\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Direction vectors mapped to the global reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "rotate_x",
        .ml_meth = (void *)pyvl_reference_frame_rotate_x,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "rotate_x(theta_x: float, /) -> Self\n"
                  "Create a copy of the frame rotated around the x-axis.\n"
                  "\n"
                  "Only works for constant orientation. Raises TypeError if orientation is time-varying.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "theta_x : float\n"
                  "    Angle by which to rotate the reference frame by.\n"
                  "Returns\n"
                  "-------\n"
                  "Self\n"
                  "    Reference frame rotated around the x-axis by the specified angle.\n",
    },
    {
        .ml_name = "rotate_y",
        .ml_meth = (void *)pyvl_reference_frame_rotate_y,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "rotate_y(theta_y: float, /) -> Self\n"
                  "Create a copy of the frame rotated around the y-axis.\n"
                  "\n"
                  "Only works for constant orientation. Raises TypeError if orientation is time-varying.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "theta_y : float\n"
                  "    Angle by which to rotate the reference frame by.\n"
                  "Returns\n"
                  "-------\n"
                  "Self\n"
                  "    Reference frame rotated around the y-axis by the specified angle.\n",
    },
    {
        .ml_name = "rotate_z",
        .ml_meth = (void *)pyvl_reference_frame_rotate_z,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "rotate_z(theta_z: float, /) -> Self\n"
                  "Create a copy of the frame rotated around the z-axis.\n"
                  "\n"
                  "Only works for constant orientation. Raises TypeError if orientation is time-varying.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "theta_z : float\n"
                  "    Angle by which to rotate the reference frame by.\n"
                  "Returns\n"
                  "-------\n"
                  "Self\n"
                  "    Reference frame rotated around the z-axis by the specified angle.\n",
    },
    {
        .ml_name = "with_offset",
        .ml_meth = (void *)pyvl_reference_frame_with_offset,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "with_offset(offset: VecLike3, /) -> ReferenceFrame\n"
                  "Create a copy of the frame with different offset value.\n"
                  "\n"
                  "Only works for constant position. Raises TypeError if position is time-varying.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "offset : VecLike3\n"
                  "    Offset to set for the reference frame relative to its parent.\n"
                  "Returns\n"
                  "-------\n"
                  "ReferenceFrame\n"
                  "    A copy of itself with the specified offset in\n"
                  "    the parent's reference frame.\n",
    },
    {
        .ml_name = "angles_from_rotation",
        .ml_meth = pyvl_matrix_to_angles,
        .ml_flags = METH_O | METH_STATIC,
        .ml_doc = "angles_from_rotation(mat: array, /) -> array\n"
                  "Compute rotation angles from a transformation matrix.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "mat : (3, 3) array\n"
                  "    Rotation matrix to convert to the rotation angles. This is done assuming that the\n"
                  "    matrix is orthogonal.\n"
                  "Returns\n"
                  "-------\n"
                  "(3,) array\n"
                  "    Rotation angles around the x-, y-, and z-axis which result in a transformation\n"
                  "    with equal rotation matrix.\n",
    },
    {
        .ml_name = "save",
        .ml_meth = (void *)pyvl_reference_frame_save,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "save(hmap: HirearchicalMap, /)\n"
                  "Serialize the ReferenceFrame into a HirearchicalMap.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "hmap : HirearchicalMap\n"
                  "    :class:`HirearchicalMap` in which to save the reference frame into.",
    },
    {
        .ml_name = "load",
        .ml_meth = (void *)pyvl_reference_frame_load,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_CLASS,
        .ml_doc = "load(hmap: HirearchicalMap, parent: ReferenceFrame | None = None/) -> Self\n"
                  "Load the ReferenceFrame from a HirearchicalMap.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "hmap : HirearchicalMap\n"
                  "    A :class:`HirearchicalMap`, which was created with a call to :meth:`ReferenceFrame.save`.\n"
                  "parent : ReferenceFrame, optional\n"
                  "    Parent of the reference frame.\n"
                  "Returns\n"
                  "-------\n"
                  "Self\n"
                  "    Deserialized :class:`ReferenceFrame`.\n",
    },
    {0},
};

PyDoc_STRVAR(pyvl_reference_frame_type_docstring,
             "ReferenceFrame(offset: VecLike3 | Callable = (0, 0, 0), theta: VecLike3 | Callable = (0, 0, 0), "
             "velocity: VecLike3 | Callable | None = None, rotation: VecLike3 | Callable | None = None, "
             "parent: ReferenceFrame | None = None)\n"
             "Class which is used to define position and orientation of geometry.\n"
             "\n"
             "Each of the position, velocity, orientation, and rotation can be either a constant vector\n"
             "or a callable with signature (float) -> (float, float, float). Callables are evaluated at\n"
             "the given time to determine the current transformation.\n"
             "\n"
             "While denoting the position of the reference frame as :math:`\\vec{r}(t)`, velocity as\n"
             ":math:`\\vec{v}(t)`, the orientation matrix as :math:`\\mathbf{T}(t)`, and its angular\n"
             "velocity as :math:`\\vec{\\omega}(t)`, the position and velocity relative to its parent,\n"
             "denoted by :math:`\\vec{r}_\\mathrm{parent}(t)` and :math:`\\vec{v}_\\mathrm{parent}(t)`,\n"
             "for a point at :math:`\\vec{r}_P` with velocity :math:`\vec{v}_P` are given by\n"
             "\n"
             ".. math::\n"
             "\n"
             "    \\vec{r}_\\mathrm{parent}(t) = \\mathbf{T}(t) \\left( \\vec{r}_P + \\vec{r}(t) \\right)\n"
             "\n"
             "and\n"
             "\n"
             ".. math::\n"
             "\n"
             "    \\vec{v}_\\mathrm{parent}(t) = \\mathbf{T}(t) \\left( \\vec{v}_P + \\vec{v} + \\vec{r}(t)\n"
             "    \\times \\vec{\\omega}(t) \\right)\n"
             "\n"
             "Of course it is also possible to transform any other quantity just with the orientation\n"
             "matrix :math:`\\mathbf{T}`.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "offset : VecLike3 or Callable, default: (0, 0, 0)\n"
             "    Position of the reference frame's origin expressed in the parent's reference frame.\n"
             "    Can be a constant vector or a callable returning the position at time t.\n"
             "\n"
             "theta : VecLike3 or Callable, default: (0, 0, 0)\n"
             "    Rotation of the reference frame relative to its parent. The rotations are applied\n"
             "    around the x, y, and z axis in that order. Can be a constant vector or a callable\n"
             "    returning the orientation (Euler angles) at time t.\n"
             "\n"
             "velocity : VecLike3 or Callable, optional, default: None\n"
             "    Linear velocity of the reference frame. Can be a constant vector or a callable\n"
             "    returning the velocity at time t. If None, velocity is assumed to be zero.\n"
             "\n"
             "rotation : VecLike3 or Callable, optional, default: None\n"
             "    Angular velocity (rotation rate) of the reference frame. Can be a constant vector\n"
             "    or a callable returning the rotation rate at time t. If None, rotation is assumed\n"
             "    to be zero.\n"
             "\n"
             "parent : ReferenceFrame, optional\n"
             "    Parent reference frame to which this frame's position and orientation are relative.\n"
             "\n");

CVL_INTERNAL
PyType_Spec pyvl_reference_frame_typespec = {
    .name = PYVL_CTYPE_NAME(ReferenceFrame),
    .basicsize = sizeof(PyVL_ReferenceFrame),
    .itemsize = 0,
    .flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_doc, (void *)pyvl_reference_frame_type_docstring},
            {Py_tp_repr, pyvl_reference_frame_repr},
            {Py_tp_getset, pyvl_reference_frame_getset},
            {Py_tp_methods, pyvl_reference_frame_methods},
            {Py_tp_new, pyvl_reference_frame_new},
            {Py_tp_dealloc, pyvl_reference_frame_dealloc},
            {Py_tp_richcompare, pyvl_reference_frame_rich_compare},
            {Py_tp_traverse, pyvl_reference_frame_traverse},
            {0, NULL},
        },
};
