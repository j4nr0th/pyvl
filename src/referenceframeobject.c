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
    return pyvl_vec_time_dependent_init(field, value, "field");
}

/**
 * Evaluate the field callable at a time and check the return has the correct type.
 *
 * @param field Callable field to call.
 * @param time_arg Time to evaluate at. Should be PyFloat.
 * @return Array of 3 doubles with the value of the field at the specified time.
 */
static bool evaluate_time_dependent_c(const pyvl_rf_time_dependent_t *field, PyObject *t, real3_t *val)
{
    return pyvl_vec_time_dependent_eval_c(field, t, val);
}

static PyObject *evaluate_time_dependent_python(const pyvl_rf_time_dependent_t *field, PyObject *t)
{
    return pyvl_vec_time_dependent_eval_py(field, t);
}

static void clear_time_dependent(pyvl_rf_time_dependent_t *field)
{
    pyvl_vec_time_dependent_clear(field);
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
    this->position.type = PYVL_VEC_CONSTANT;
    this->velocity.type = PYVL_VEC_CONSTANT;
    this->orientation.type = PYVL_VEC_CONSTANT;
    this->rotation.type = PYVL_VEC_CONSTANT;

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
    if (this->position.type == PYVL_VEC_CALLABLE)
        Py_VISIT(this->position.value.callable);
    if (this->velocity.type == PYVL_VEC_CALLABLE)
        Py_VISIT(this->velocity.value.callable);
    if (this->orientation.type == PYVL_VEC_CALLABLE)
        Py_VISIT(this->orientation.value.callable);
    if (this->rotation.type == PYVL_VEC_CALLABLE)
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
    const char *pos_str = this->position.type == PYVL_VEC_CALLABLE ? "<callable>" : "(constant)";
    const char *ori_str = this->orientation.type == PYVL_VEC_CALLABLE ? "<callable>" : "(constant)";
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

/**
 * Compute the number of ancestors a reference frame has.
 *
 * @param this Reference frame for which the depth is computed.
 * @return Number of ancestors of the reference frame.
 */
static unsigned reference_frame_depth(const PyVL_ReferenceFrame *this)
{
    unsigned parent_cnt = 0;
    for (const PyVL_ReferenceFrame *p = this->parent; p; p = p->parent)
        parent_cnt += 1;
    return parent_cnt;
}

static PyObject *pyvl_reference_frame_get_parents(PyObject *self, void *Py_UNUSED(closure))
{
    const PyVL_ReferenceFrame *this = (PyVL_ReferenceFrame *)self;
    const unsigned parent_cnt = reference_frame_depth(this);
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
        if (this->position.type == PYVL_VEC_CONSTANT)
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
        if (this->orientation.type == PYVL_VEC_CONSTANT)
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

static PyObject *pyvl_reference_frame_get_is_moving(PyObject *self, void *Py_UNUSED(closure))
{
    for (const PyVL_ReferenceFrame *rf = (PyVL_ReferenceFrame *)self; rf; rf = rf->parent)
    {
        // If either velocity or rotation are not constant and zero, it is moving.
        if (!(rf->velocity.type == PYVL_VEC_CONSTANT && real3_all_zero(rf->velocity.value.constant)) ||
            !(rf->rotation.type == PYVL_VEC_CONSTANT && real3_all_zero(rf->rotation.value.constant)))
            Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
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
    {
        .name = "is_moving",
        .get = pyvl_reference_frame_get_is_moving,
        .doc = "bool : True if either the reference frame or its ancestors are moving.\n",
    },
    {0},
};

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

typedef bool (*const rf_iter_func)(const PyVL_ReferenceFrame *rf, void *param);

/**
 * Callback and parameter used to iterate over reference frames.
 */
typedef struct
{
    rf_iter_func iter_func;
    void *param;
} rf_iter_callback_t;

/**
 * Iterate from one reference frame to its ancestor, starting at "start" and not including "end".
 *
 * @param start Reference frame where the iteration starts.
 * @param end Reference frame where the iteration stops.
 * @param callback Iteration callback called for each reference frame.
 * @return False if "start" is not a descended of "end" or if the iteration function returns false.
 */
static bool reference_frame_iterate_down_to_ancestor(const PyVL_ReferenceFrame *start, const PyVL_ReferenceFrame *end,
                                                     const rf_iter_callback_t callback)
{
    for (const PyVL_ReferenceFrame *this = start; this != end; this = this->parent)
    {
        if (!this)
        {
            PyErr_SetString(PyExc_ValueError,
                            "The input source reference frame is not a descendant of the destination reference frame.");
            return false;
        }
        // Call iteration function
        if (!callback.iter_func(this, callback.param))
            return false;
    }

    return true;
}

/**
 * Call the iteration on each iteration from "start" to "end" (including "start" and excluding "end") in
 * reversed order.
 *
 * @param start Reference frame from which at which to start iterating (never NULL).
 * @param end The descendant of "start" at which to end the iteration.
 * @param callback Iteration callback called for each reference frame.
 * @return False if "end" is not a descended of "start" or if the iteration function returns false.
 */
static inline bool reference_frame_iterate_up_to_descendant(const PyVL_ReferenceFrame *const start,
                                                            const PyVL_ReferenceFrame *const end,
                                                            const rf_iter_callback_t callback)
{
    const PyVL_ReferenceFrame *const parent = end->parent;
    if (parent != start)
    {
        if (!parent)
        {
            PyErr_SetString(PyExc_ValueError,
                            "The specified end reference frame was not a descendant of the starting reference frame.");
            return false;
        }
        const bool res = reference_frame_iterate_up_to_descendant(start, parent, callback);
        if (!res)
            return res;
    }

    if (!callback.iter_func(end, callback.param))
        return false;

    return true;
}

/**
 * Arguments for transformation of either position and direction vectors.
 */
typedef struct
{
    PyObject *t;  // Python time object.
    size_t n;     // Number of vectors to transform.
    real3_t *out; // In-Out where transformed vectors are read from, transformed, and written.
    bool inv;     // Inverse transformation.
} transformation_args_t;

static inline bool transform_position_parent(const PyVL_ReferenceFrame *this, void *param)
{
    const transformation_args_t *const args = param;
    real3_t angles;
    if (!evaluate_time_dependent_c(&this->orientation, args->t, &angles))
        return false;

    // Transformation matrix
    const real3x3_t mat = real3x3_from_angles(angles);

    real3_t off;
    if (!evaluate_time_dependent_c(&this->position, args->t, &off))
        return false;

    for (size_t i = 0; i < args->n; ++i)
    {
        args->out[i] = !args->inv ? real3_add(real3x3_vecmul(mat, args->out[i]), off)
                                  : real3x3_vecmul_transpose(mat, real3_sub(args->out[i], off));
    }

    return true;
}

static inline bool transform_vector_parent(const PyVL_ReferenceFrame *this, void *param)
{
    const transformation_args_t *const args = param;
    real3_t angles;
    if (!evaluate_time_dependent_c(&this->orientation, args->t, &angles))
        return false;

    // Transformation matrix
    const real3x3_t mat = real3x3_from_angles(angles);

    for (size_t i = 0; i < args->n; ++i)
    {
        args->out[i] = !args->inv ? real3x3_vecmul(mat, args->out[i]) : real3x3_vecmul_transpose(mat, args->out[i]);
    }

    return true;
}

/**
 * Arguments for transformation of both position and velocity vectors.
 */
typedef struct
{
    PyObject *t;  // Python time object.
    size_t n;     // Number of vectors to transform.
    real3_t *vel; // Array where transformed velocity vectors are written.
    real3_t *pos; // Array where transformed position vectors are written.
    bool inv;     // Perform the inverse transformation instead.
} transformation_vel_pos_args_t;

static bool transform_vel_pos_parent(const PyVL_ReferenceFrame *this, void *param)
{
    const transformation_vel_pos_args_t *const args = param;

    real3_t angles;
    real3_t offset;
    real3_t linear_velocity;
    real3_t rotation_rate;
    if (!evaluate_time_dependent_c(&this->orientation, args->t, &angles) ||
        !evaluate_time_dependent_c(&this->position, args->t, &offset) ||
        !evaluate_time_dependent_c(&this->velocity, args->t, &linear_velocity) ||
        !evaluate_time_dependent_c(&this->rotation, args->t, &rotation_rate))
        return false;

    const real3x3_t mat = real3x3_from_angles(angles);

#pragma omp simd
    for (size_t i = 0; i < args->n; ++i)
    {
        real3_t pos = args->pos[i];
        real3_t vel = args->vel[i];
        if (!args->inv)
        {
            if (!real3_all_zero(angles))
            {
                // Transform to parent coordinate system
                pos = real3x3_vecmul(mat, pos);
                vel = real3x3_vecmul(mat, vel);
            }
            // Compute rotation velocity
            if (!real3_all_zero(rotation_rate))
                vel = real3_add(vel, real3_cross(pos, rotation_rate));

            // Add offsets relative to parent
            if (!real3_all_zero(offset))
                pos = real3_add(pos, offset);

            if (!real3_all_zero(linear_velocity))
                vel = real3_add(vel, linear_velocity);
        }
        else
        {
            // Move to child's origin
            if (!real3_all_zero(offset))
                pos = real3_sub(pos, offset);
            if (!real3_all_zero(linear_velocity))
                vel = real3_sub(vel, linear_velocity);

            // Compute rotation velocity
            if (!real3_all_zero(rotation_rate))
                vel = real3_sub(vel, real3_cross(pos, rotation_rate));

            // Transform with transpose of the matrix (velocity also needs to remove rotation)
            if (!real3_all_zero(angles))
            {
                pos = real3x3_vecmul_transpose(mat, pos);
                vel = real3x3_vecmul_transpose(mat, vel);
            }
        }
        args->pos[i] = pos;
        args->vel[i] = vel;
    }

    return true;
}

/**
 * Finds the first ancestor of the two reference frames.
 *
 * @param r1 First reference frame.
 * @param r2 Second reference frame.
 * @return First common ancestor of the two reference frames. NULL means the global reference frame.
 */
static inline const PyVL_ReferenceFrame *nearest_common_ancestor(const PyVL_ReferenceFrame *r1,
                                                                 const PyVL_ReferenceFrame *r2)
{
    // If either has no parent or is NULL, then their common ancestor is global (NULL)
    if (!r1 || !r2 || !r1->parent || !r2->parent)
        return NULL;

    // Search from p1 to root
    for (const PyVL_ReferenceFrame *p1 = r1; p1; p1 = p1->parent)
    {
        // Search form p2 to root
        for (const PyVL_ReferenceFrame *p2 = r2; p2; p2 = p2->parent)
        {
            // Common ancestor found
            if (p2 == p1)
                return p1;
        }
    }

    return NULL;
}

/**
 * Iterate from a reference frame to a common ancestor, then to the end reference frame.
 *
 * @param start Initial reference frame where the iteration starts.
 * @param end Final reference frame where the iteration ends.
 * @param callback_down Callback used while iterating from the start to the common ancestor.
 * @param callback_up Callback used while iterating from the common ancestor to the end.
 * @return On failure false.
 */
static inline bool reference_frame_iterate_between_reference_frames(const PyVL_ReferenceFrame *start,
                                                                    const PyVL_ReferenceFrame *end,
                                                                    const rf_iter_callback_t callback_down,
                                                                    const rf_iter_callback_t callback_up)
{
    // Get the common ancestor
    const PyVL_ReferenceFrame *const common = nearest_common_ancestor(start, end);

    // Iterate from start to common (if start is not already NULL)
    if (start && !reference_frame_iterate_down_to_ancestor(start, common, callback_down))
        return false;

    // Iterate from common to end (if end is not already NULL)
    if (end && !reference_frame_iterate_up_to_descendant(common, end, callback_up))
        return false;

    return true;
}

static inline PyArrayObject *pyvl_ensure_output_array_as_input(PyArrayObject *out_array, const PyArrayObject *in_array)
{
    const unsigned ndim = PyArray_NDIM(in_array);
    const npy_intp *const dims = PyArray_DIMS(in_array);
    if (!out_array)
    {
        return (PyArrayObject *)PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    }

    if (check_input_array(out_array, ndim, dims, NPY_DOUBLE,
                          NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "out") < 0)
        return NULL;
    Py_INCREF(out_array);
    return out_array;
}

static inline PyObject *reference_frame_process_transformation(const PyVL_ReferenceFrame *start,
                                                               const PyVL_ReferenceFrame *end, const double t,
                                                               PyObject *in_any, PyArrayObject *out_array,
                                                               const rf_iter_func trans_func)
{
    PyArrayObject *const in_array =
        (PyArrayObject *)PyArray_FROMANY(in_any, NPY_DOUBLE, 1, INT_MAX, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!in_array)
    {
        return NULL;
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
        return NULL;
    }
    out_array = pyvl_ensure_output_array_as_input(out_array, in_array);
    if (!out_array)
    {
        Py_DECREF(in_array);
        return NULL;
    }
    size_t n_entries = 1;
    for (unsigned i = 0; i < dim_in - 1; i++)
        n_entries *= dims_in[i];

    const real3_t *const p_in = PyArray_DATA(in_array);
    real3_t *const p_out = PyArray_DATA(out_array);
    if (in_array != out_array)
        for (size_t i = 0; i < n_entries; ++i)
        {
            p_out[i] = p_in[i];
        }
    Py_DECREF(in_array);
    if (start == end)
        return (PyObject *)out_array;

    PyObject *const time = PyFloat_FromDouble(t);
    if (!time)
    {
        Py_DECREF(out_array);
        return NULL;
    }

    transformation_args_t args_down = (transformation_args_t){.t = time, .n = n_entries, .out = p_out, .inv = false};
    transformation_args_t args_up = args_down;
    args_up.inv = true;
    const rf_iter_callback_t callback_backward = {.iter_func = trans_func, .param = &args_down};
    const rf_iter_callback_t callback_forward = {.iter_func = trans_func, .param = &args_up};
    const bool res = reference_frame_iterate_between_reference_frames(start, end, callback_backward, callback_forward);
    Py_DECREF(time);
    if (!res)
    {
        Py_DECREF(out_array);
        out_array = NULL;
    }

    return (PyObject *)out_array;
}

/**
 * Perform a transformation on a set of input vectors based between two reference frames.
 *
 * NOTE: Apparently having the ``inline`` keyword really does help convince GCC to inline this.
 *
 * @param start Initial reference frame where the quantities are initially.
 * @param end Final reference frame where the quantities are transferred to.
 * @param args Positional arguments passed to the method.
 * @param nargs The number of positional arguments.
 * @param kwnames Keyword names passed to the method; can be NULL.
 * @param trans_func Callback with function that specifies the transformation.
 * @return A NumPy array containing the transformed vectors on success, or NULL on failure (with a Python exception
 * raised).
 */
static inline PyObject *reference_frame_transformation_method(const PyVL_ReferenceFrame *start,
                                                              const PyVL_ReferenceFrame *end, PyObject *const *args,
                                                              const Py_ssize_t nargs, const PyObject *kwnames,
                                                              const rf_iter_func trans_func)
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
        return NULL;

    return reference_frame_process_transformation(start, end, t, in_any, out_array, trans_func);
}

static inline PyObject *reference_frame_process_velocity(const PyVL_ReferenceFrame *start,
                                                         const PyVL_ReferenceFrame *end, const double t,
                                                         PyObject *pos_any, PyObject *vel_any, PyArrayObject *out_pos,
                                                         PyArrayObject *out_vel)
{
    PyArrayObject *const vel_array =
        (PyArrayObject *)PyArray_FROMANY(vel_any, NPY_DOUBLE, 1, INT_MAX, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!vel_array)
    {
        return NULL;
    }
    const npy_intp dim_in = PyArray_NDIM(vel_array);
    const npy_intp *dims_in = PyArray_DIMS(vel_array);
    if (dims_in[dim_in - 1] != 3)
    {
        PyErr_Format(PyExc_ValueError,
                     "Input array does not have the last axis with 3 dimensions "
                     "(shape is (..., %u) instead of (..., 3)).",
                     (unsigned)dims_in[dim_in - 1]);
        Py_DECREF(vel_array);
        return NULL;
    }

    PyArrayObject *const pos_array = (PyArrayObject *)PyArray_FROMANY(
        pos_any, NPY_DOUBLE, dim_in, dim_in, NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!pos_array)
    {
        Py_DECREF(vel_array);
        return NULL;
    }
    if (check_input_array(pos_array, dim_in, dims_in, NPY_DOUBLE, 0, "position array") < 0)
    {
        Py_DECREF(pos_array);
        Py_DECREF(vel_array);
        return NULL;
    }

    out_vel = pyvl_ensure_output_array_as_input(out_vel, vel_array);
    out_pos = pyvl_ensure_output_array_as_input(out_pos, pos_array);
    if (!out_vel || !out_pos)
    {
        Py_XDECREF(out_pos);
        Py_XDECREF(out_vel);
        Py_DECREF(pos_array);
        Py_DECREF(vel_array);
        return NULL;
    }

    size_t n_entries = 1;
    for (unsigned i = 0; i < dim_in - 1; i++)
        n_entries *= dims_in[i];

    const real3_t *const in_vel = PyArray_DATA(vel_array);
    const real3_t *const in_pos = PyArray_DATA(pos_array);
    real3_t *const out_v = PyArray_DATA(out_vel);
    real3_t *const out_p = PyArray_DATA(out_pos);
    if (vel_array != out_vel)
        for (size_t i = 0; i < n_entries; ++i)
            out_v[i] = in_vel[i];
    Py_DECREF(vel_array);

    if (pos_array != out_pos)
        for (size_t i = 0; i < n_entries; ++i)
            out_p[i] = in_pos[i];
    Py_DECREF(pos_array);

    if (start != end)
    {
        PyObject *const time = PyFloat_FromDouble(t);
        if (!time)
        {
            Py_DECREF(out_pos);
            Py_DECREF(out_vel);
            return NULL;
        }
        transformation_vel_pos_args_t args_down =
            (transformation_vel_pos_args_t){.t = time, .n = n_entries, .vel = out_v, .pos = out_p, .inv = false};
        transformation_vel_pos_args_t args_up = args_down;
        args_up.inv = true;
        const rf_iter_callback_t callback_backward = {.iter_func = transform_vel_pos_parent, .param = &args_down};
        const rf_iter_callback_t callback_forward = {.iter_func = transform_vel_pos_parent, .param = &args_up};
        const bool res =
            reference_frame_iterate_between_reference_frames(start, end, callback_backward, callback_forward);
        Py_DECREF(time);
        if (!res)
        {
            Py_DECREF(out_vel);
            Py_DECREF(out_pos);
            return NULL;
        }
    }

    // Pack the output into a tuple and return
    PyObject *const out =
        cpyutl_output_create_check(CPYOUT_TYPE_TUPLE, (cpyutl_output_t[]){
                                                          {.type = CPYOUT_TYPE_PYOBJ, .value_obj = (PyObject *)out_pos},
                                                          {.type = CPYOUT_TYPE_PYOBJ, .value_obj = (PyObject *)out_vel},
                                                          {0},
                                                      });

    // If tuple was constructed, their ref counts were increased.
    Py_DECREF(out_vel);
    Py_DECREF(out_pos);
    // Return the tuple
    return out;
}

/**
 * Perform a transformation on a set of input vectors based between two reference frames.
 *
 * NOTE: Apparently having the ``inline`` keyword really does help convince GCC to inline this.
 *
 * @param start Initial reference frame where the quantities are initially.
 * @param end Final reference frame where the quantities are transferred to.
 * @param args Positional arguments passed to the method.
 * @param nargs The number of positional arguments.
 * @param kwnames Keyword names passed to the method; can be NULL.
 * @return A NumPy array containing the transformed vectors on success, or NULL on failure (with a Python exception
 * raised).
 */
static inline PyObject *reference_frame_transformation_velocity(const PyVL_ReferenceFrame *start,
                                                                const PyVL_ReferenceFrame *end, PyObject *const *args,
                                                                const Py_ssize_t nargs, const PyObject *kwnames)
{
    PyObject *vel_any, *pos_any;
    PyArrayObject *out_vel = NULL, *out_pos = NULL;
    double t = 0.0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&pos_any, .kwname = "position"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&vel_any, .kwname = "velocity"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .kwname = "time", .optional = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&out_pos,
                 .kwname = "out_position",
                 .type_check = &PyArray_Type,
                 .optional = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&out_vel,
                 .kwname = "out_velocity",
                 .type_check = &PyArray_Type,
                 .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    return reference_frame_process_velocity(start, end, t, pos_any, vel_any, out_pos, out_vel);
}

static PyObject *pyvl_reference_frame_from_parent_position(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, const Py_ssize_t nargs,
                                                           const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_method(this->parent, this, args, nargs, kwnames, transform_position_parent);
}

static PyObject *pyvl_reference_frame_from_parent_velocity(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, const Py_ssize_t nargs,
                                                           const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_velocity(this->parent, this, args, nargs, kwnames);
}

static PyObject *pyvl_reference_frame_from_parent_vector(PyObject *self, PyTypeObject *defining_class,
                                                         PyObject *const *args, const Py_ssize_t nargs,
                                                         const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_method(this->parent, this, args, nargs, kwnames, transform_vector_parent);
}

static PyObject *pyvl_reference_frame_to_parent_position(PyObject *self, PyTypeObject *defining_class,
                                                         PyObject *const *args, const Py_ssize_t nargs,
                                                         const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_method(this, this->parent, args, nargs, kwnames, transform_position_parent);
}

static PyObject *pyvl_reference_frame_to_parent_velocity(PyObject *self, PyTypeObject *defining_class,
                                                         PyObject *const *args, const Py_ssize_t nargs,
                                                         const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_velocity(this, this->parent, args, nargs, kwnames);
}

static PyObject *pyvl_reference_frame_to_parent_vector(PyObject *self, PyTypeObject *defining_class,
                                                       PyObject *const *args, const Py_ssize_t nargs,
                                                       const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_method(this, this->parent, args, nargs, kwnames, transform_vector_parent);
}

static PyObject *pyvl_reference_frame_from_global_position(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, const Py_ssize_t nargs,
                                                           const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_method(NULL, this, args, nargs, kwnames, transform_position_parent);
}

static PyObject *pyvl_reference_frame_from_global_velocity(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, const Py_ssize_t nargs,
                                                           PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_velocity(NULL, this, args, nargs, kwnames);
}

static PyObject *pyvl_reference_frame_from_global_vector(PyObject *self, PyTypeObject *defining_class,
                                                         PyObject *const *args, const Py_ssize_t nargs,
                                                         PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_method(NULL, this, args, nargs, kwnames, transform_vector_parent);
}

static PyObject *pyvl_reference_frame_to_global_position(PyObject *self, PyTypeObject *defining_class,
                                                         PyObject *const *args, const Py_ssize_t nargs,
                                                         const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_method(this, NULL, args, nargs, kwnames, transform_position_parent);
}

static PyObject *pyvl_reference_frame_to_global_velocity(PyObject *self, PyTypeObject *defining_class,
                                                         PyObject *const *args, const Py_ssize_t nargs,
                                                         PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_velocity(this, NULL, args, nargs, kwnames);
}

static PyObject *pyvl_reference_frame_to_global_vector(PyObject *self, PyTypeObject *defining_class,
                                                       PyObject *const *args, const Py_ssize_t nargs,
                                                       const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    return reference_frame_transformation_method(this, NULL, args, nargs, kwnames, transform_vector_parent);
}

static PyObject *pyvl_reference_frame_transform_position(PyTypeObject *class, PyObject *const *args,
                                                         const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *const state = get_module_state(class);
    if (!state)
        return NULL;

    const PyVL_ReferenceFrame *src = NULL, *dst = NULL;
    PyObject *in;
    double t = 0.0;
    PyArrayObject *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&in, .kwname = "x"},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&src,
                 .kwname = "src",
                 .type_check = state->rf_type,
                 .optional = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&dst,
                 .kwname = "dst",
                 .type_check = state->rf_type,
                 .optional = true},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .kwname = "time", .optional = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)out,
                 .kwname = "out",
                 .optional = true,
                 .type_check = &PyArray_Type},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    return reference_frame_process_transformation(src, dst, t, in, out, transform_position_parent);
}

static PyObject *pyvl_reference_frame_transform_vector(PyTypeObject *class, PyObject *const *args,
                                                       const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *const state = get_module_state(class);
    if (!state)
        return NULL;

    const PyVL_ReferenceFrame *src = NULL, *dst = NULL;
    PyObject *in;
    double t = 0.0;
    PyArrayObject *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&in, .kwname = "x"},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&src,
                 .kwname = "src",
                 .type_check = state->rf_type,
                 .optional = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&dst,
                 .kwname = "dst",
                 .type_check = state->rf_type,
                 .optional = true},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .kwname = "time", .optional = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)out,
                 .kwname = "out",
                 .optional = true,
                 .type_check = &PyArray_Type},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    return reference_frame_process_transformation(src, dst, t, in, out, transform_vector_parent);
}

static PyObject *pyvl_reference_frame_transform_vel_pos(PyTypeObject *class, PyObject *const *args,
                                                        const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *const state = get_module_state(class);
    if (!state)
        return NULL;

    const PyVL_ReferenceFrame *src, *dst;
    PyObject *in_pos, *in_vel;
    double t = 0.0;
    PyArrayObject *out_pos = NULL, *out_vel = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&src, .kwname = "src", .type_check = state->rf_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&dst, .kwname = "dst", .type_check = state->rf_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&in_pos, .kwname = "position"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&in_vel, .kwname = "velocity"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t, .kwname = "time", .optional = true},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)out_pos,
                 .kwname = "out_position",
                 .optional = true,
                 .type_check = &PyArray_Type},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)out_vel,
                 .kwname = "out_velocity",
                 .optional = true,
                 .type_check = &PyArray_Type},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    return reference_frame_process_velocity(src, dst, t, in_pos, in_vel, out_pos, out_vel);
}

static PyObject *pyvl_reference_frame_rotate_x(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                               const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    if (this->orientation.type == PYVL_VEC_CALLABLE)
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
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    if (this->orientation.type == PYVL_VEC_CALLABLE)
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
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    if (this->orientation.type == PYVL_VEC_CALLABLE)
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
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    if (this->position.type == PYVL_VEC_CALLABLE)
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

    PyArrayObject *const off_array =
        (PyArrayObject *)PyArray_FromAny(in_any, PyArray_DescrFromType(NPY_DOUBLE), 1, 1, NPY_ARRAY_C_CONTIGUOUS, NULL);
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
    new->position.type = PYVL_VEC_CONSTANT;
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

static bool pyvl_rf_serialize_entry(const pyvl_rf_time_dependent_t *entry, const char *key, PyObject *hmap,
                                    PyObject *serializer)
{
    return pyvl_vec_time_dependent_serialize(entry, key, hmap, serializer);
}

static PyObject *pyvl_reference_frame_save(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                           const Py_ssize_t nargs, const PyObject *kwnames)
{
    const PyVL_ReferenceFrame *this;
    const module_state_t *state;
    if (!ensure_rf_and_state(self, defining_class, &this, &state))
        return NULL;

    PyObject *hmap, *serializer_callable;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&hmap, .kwname = "hmap"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&serializer_callable, .kwname = "serializer"},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (!PyMapping_Check(hmap))
    {
        PyErr_Format(PyExc_TypeError, "The input parameter is not a mapping.");
        return NULL;
    }
    if (!PyCallable_Check(serializer_callable))
    {
        PyErr_Format(PyExc_TypeError, "Serializer is not a callable.");
        return NULL;
    }

    if (!pyvl_rf_serialize_entry(&this->orientation, "orientation", hmap, serializer_callable) ||
        !pyvl_rf_serialize_entry(&this->velocity, "velocity", hmap, serializer_callable) ||
        !pyvl_rf_serialize_entry(&this->position, "position", hmap, serializer_callable) ||
        !pyvl_rf_serialize_entry(&this->rotation, "rotation", hmap, serializer_callable))
        return NULL;

    Py_RETURN_NONE;
}

static bool pyvl_rf_deserialize_entry(const char *key, PyObject *hmap, PyObject *deserializer,
                                      pyvl_rf_time_dependent_t *entry)
{
    return pyvl_vec_time_dependent_deserialize(key, hmap, deserializer, entry);
}

static PyObject *pyvl_reference_frame_load(PyTypeObject *type, PyObject *const *args, const Py_ssize_t nargs,
                                           const PyObject *kwnames)
{
    const module_state_t *const state = get_module_state(type);
    if (!state)
        return NULL;

    PyObject *group, *deserializer;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&group, .kwname = "group"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&deserializer, .kwname = "deserializer"},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (!PyMapping_Check(group))
    {
        PyErr_Format(PyExc_TypeError, "The input parameter is not a mapping.");
        return NULL;
    }
    if (!PyCallable_Check(deserializer))
    {
        PyErr_Format(PyExc_TypeError, "Deserializer is not a callable.");
        return NULL;
    }

    PyVL_ReferenceFrame *const this = (PyVL_ReferenceFrame *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;

    // Clear it up
    this->position = (pyvl_rf_time_dependent_t){0};
    this->orientation = (pyvl_rf_time_dependent_t){0};
    this->velocity = (pyvl_rf_time_dependent_t){0};
    this->rotation = (pyvl_rf_time_dependent_t){0};
    this->parent = NULL;

    if (!pyvl_rf_deserialize_entry("position", group, deserializer, &this->position) ||
        !pyvl_rf_deserialize_entry("velocity", group, deserializer, &this->velocity) ||
        !pyvl_rf_deserialize_entry("orientation", group, deserializer, &this->orientation) ||
        !pyvl_rf_deserialize_entry("rotation", group, deserializer, &this->rotation))
    {
        Py_DECREF(this);
        return NULL;
    }

    // Deserialize the parent
    PyObject *parent_group;
    int res;
    if ((res = PyMapping_GetOptionalItemString(group, "parent", &parent_group)))
    {
        if (res < 0)
        {
            // Some weird error happened
            Py_DECREF(this);
            return NULL;
        }

        if (!PyObject_TypeCheck(parent_group, Py_TYPE(group)))
        {
            PyErr_Format(PyExc_TypeError, "Parent group is not a %s, but is %s.", Py_TYPE(group)->tp_name,
                         Py_TYPE(parent_group)->tp_name);
            Py_DECREF(this);
            Py_DECREF(parent_group);
            return NULL;
        }

        PyVL_ReferenceFrame *parent = (PyVL_ReferenceFrame *)pyvl_reference_frame_load(
            type, (PyObject *[2]){parent_group, deserializer}, 2, NULL);
        Py_DECREF(parent_group);
        if (!parent)
        {
            Py_DECREF(this);
            return NULL;
        }

        this->parent = parent;
    }

    return (PyObject *)this;
}

static bool ancestor_position_transform(const PyVL_ReferenceFrame *const ancestor,
                                        const PyVL_ReferenceFrame *const this, real3x3_t *mat, real3_t *off,
                                        PyObject *time_obj)
{
    const PyVL_ReferenceFrame *current = this;
    while (current != ancestor)
    {
        if (current == NULL)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Could not get the position transformation from reference frame %p to %p, as %p was not the "
                         "ancestor of %p.",
                         this, ancestor, ancestor, this);
            return false;
        }

        // Get the transformation to the parent
        real3_t rel_off;
        real3_t rel_ori;
        if (!evaluate_time_dependent_c(&current->orientation, time_obj, &rel_off) ||
            !evaluate_time_dependent_c(&current->position, time_obj, &rel_ori))
            return false;

        const real3x3_t rel_mat = real3x3_from_angles(rel_ori);

        // Apply the transformation to the ones we have so far
        *off = real3_add(real3x3_vecmul(rel_mat, *off), rel_off);
        *mat = real3x3_matmul(rel_mat, *mat);

        current = current->parent;
    }

    return true;
}

static bool relative_transform(const PyVL_ReferenceFrame *ancestor, const PyVL_ReferenceFrame *this,
                               const PyVL_ReferenceFrame *that, PyObject *time_obj, real3x3_t *p_mat, real3_t *p_off)
{
    // Initializers
    const real3x3_t mat = {.m00 = 1, .m11 = 1, .m22 = 1};
    const real3_t off = {0};

    // Init transforms relative to ancestor
    real3_t off_this = off, off_that = off;
    real3x3_t mat_this = mat, mat_that = mat;

    // Get the transforms (if not already at the ancestor)
    if (!(this == ancestor || ancestor_position_transform(ancestor, this, &mat_this, &off_this, time_obj)) ||
        !(that == ancestor || ancestor_position_transform(ancestor, that, &mat_that, &off_that, time_obj)))
        return false;

    // Make these relative to one another
    // Relative orientation matrix
    *p_mat = real3x3_matmul_transpose(mat_this, mat_that);
    // Relative offset
    *p_off = real3_sub(off_this, real3x3_vecmul(*p_mat, off_that));

    return true;
}

static const PyVL_ReferenceFrame *nearest_non_constant_reference_frame(const PyVL_ReferenceFrame *current,
                                                                       const PyVL_ReferenceFrame *const ancestor)
{
    const PyVL_ReferenceFrame *used = ancestor;
    while (current != ancestor)
    {
        // Can we use the current one?
        if (current->orientation.type != PYVL_VEC_CONSTANT || current->position.type != PYVL_VEC_CONSTANT)
            used = current->parent;

        // Go down the hierarchy towards this
        current = current->parent;
    }
    return used;
}

static PyObject *pyvl_reference_frame_moved_relative_to(PyObject *self, PyTypeObject *defining_class,
                                                        PyObject *const *args, const Py_ssize_t nargs,
                                                        const PyObject *kwnames)
{
    const module_state_t *mod_state;
    const PyVL_ReferenceFrame *this;
    if (!ensure_rf_and_state(self, defining_class, &this, &mod_state))
        return NULL;

    const PyVL_ReferenceFrame *that;
    double t_start, t_end;
    double tol;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&that, .kwname = "other"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t_start, .kwname = "t_start"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &t_end, .kwname = "t_end"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &tol, .kwname = "tol"},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (tol < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Tolerance cannot be less than 0.");
        return NULL;
    }

    if (Py_IsNone((PyObject *)that))
    {
        // Global RF
        that = NULL;
    }
    else if (!PyObject_TypeCheck(that, mod_state->rf_type))
    {
        // Not a RF, error
        PyErr_Format(PyExc_TypeError, "other must be a %s, but was %s.", mod_state->rf_type->tp_name,
                     Py_TYPE(that)->tp_name);
        return NULL;
    }

    const PyVL_ReferenceFrame *ancestor = nearest_common_ancestor(this, that);
    // Keep on moving down the hierarchy until we either reach the ancestor or a RF that is not moving constantly
    while (this != ancestor)
    {
        if (this->orientation.type != PYVL_VEC_CONSTANT || this->position.type != PYVL_VEC_CONSTANT)
            break;
        this = this->parent;
    }
    while (that != ancestor)
    {
        if (that->orientation.type != PYVL_VEC_CONSTANT || that->position.type != PYVL_VEC_CONSTANT)
            break;
        that = that->parent;
    }

    if (this == that)
        // Both reached the ancestor without any non-constant position and orientation, so they did not move
        Py_RETURN_FALSE;

    if (this == ancestor)
    {
        // We can move ancestor closer to that.
        ancestor = this = nearest_non_constant_reference_frame(that, this);
    }
    else if (that == ancestor)
    {
        // We can move ancestor closer to this
        ancestor = that = nearest_non_constant_reference_frame(this, that);
    }

    // We have to evaluate the position and orientation for both before and after
    real3_t rel_off_start, rel_off_end;
    real3x3_t rel_mat_start, rel_mat_end;
    rel_off_start = rel_off_end = (real3_t){0};
    rel_mat_start = rel_mat_end = (real3x3_t){.m00 = 1, .m11 = 1, .m22 = 1};

    PyObject *start_time = NULL, *end_time = NULL;
    if ((start_time = PyFloat_FromDouble(t_start)) == NULL || (end_time = PyFloat_FromDouble(t_end)) == NULL)
    {
        Py_XDECREF(start_time);
        Py_XDECREF(end_time);
        return NULL;
    }

    // Compute relative transform
    const bool managed_transform =
        (relative_transform(ancestor, this, that, start_time, &rel_mat_start, &rel_off_start) &&
         relative_transform(ancestor, this, that, end_time, &rel_mat_end, &rel_off_end));

    Py_DECREF(start_time);
    Py_DECREF(end_time);
    if (!managed_transform)
    {
        // Something went wrong, so we return.
        return NULL;
    }

    // Now check how these two transforms compare
    const real3_t diff_offset = real3_sub(rel_off_end, rel_off_start);
    if (real3_mag(diff_offset) > tol)
        // The offsets are too different
        Py_RETURN_TRUE;

    // Relative rotation angles between the transforms
    const real3_t diff_orient = angles_from_real3x3(real3x3_matmul_transpose(rel_mat_end, rel_mat_start));
    if (fabs(diff_orient.x) > tol || fabs(diff_orient.y) > tol || fabs(diff_orient.z) > tol)
        Py_RETURN_TRUE;

    Py_RETURN_FALSE;
}

static PyObject *pyvl_reference_frame_common_ancestor(PyObject *self, PyTypeObject *defining_class,
                                                      PyObject *const *args, const Py_ssize_t nargs,
                                                      const PyObject *kwnames)
{
    const module_state_t *mod_state;
    const PyVL_ReferenceFrame *this;
    if (!ensure_rf_and_state(self, defining_class, &this, &mod_state))
        return NULL;

    const PyVL_ReferenceFrame *that;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&that, .kwname = "other"},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (Py_IsNone((PyObject *)that))
        Py_RETURN_NONE;

    if (!PyObject_TypeCheck(that, mod_state->rf_type))
    {
        PyErr_Format(PyExc_TypeError, "other must be a %s, but was %s.", mod_state->rf_type->tp_name,
                     Py_TYPE(that)->tp_name);
        return NULL;
    }

    const PyVL_ReferenceFrame *ancestor = nearest_common_ancestor(this, that);
    if (ancestor == NULL)
        Py_RETURN_NONE;

    Py_INCREF(ancestor);
    return (PyObject *)ancestor;
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
    // RF <-> Parent
    {
        .ml_name = "from_parent_position",
        .ml_meth = (void *)pyvl_reference_frame_from_parent_position,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_parent_position,(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map position vectors from parent reference frame to the child reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in parent reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the child reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "from_parent_velocity",
        .ml_meth = (void *)pyvl_reference_frame_from_parent_velocity,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_parent_velocity(position: array, velocity: array, time: float = 0.0, out_position: out_array | "
                  "None = None, out_velocity: out_array | None = None) -> tuple[out_array, out_array]\n"
                  "Map velocity vectors from parent reference frame to the local reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "position : (N, 3) array\n"
                  "    Array of :math:`N` position vectors in :math:`\\mathbb{R}^3` in parent reference frame.\n"
                  "\n"
                  "velocity : (N, 3) array\n"
                  "    Array of :math:`N` velocity vectors in :math:`\\mathbb{R}^3` in parent reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out_position : (N, 3) array, optional\n"
                  "    Array which receives the mapped position vectors. Must have the exact shape of ``position``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "out_velocity : (N, 3) array, optional\n"
                  "    Array which receives the mapped velocity vectors. Must have the exact shape of ``velocity``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the local reference frame. If the ``out_position`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out_position`` was not specified,\n"
                  "    then a new array will be allocated."
                  "\n"
                  "(N, 3) array\n"
                  "    Velocity vectors mapped to the local reference frame. If the ``out_velocity`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out_velocity`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "from_parent_vector",
        .ml_meth = (void *)pyvl_reference_frame_from_parent_vector,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_parent_vector(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map vectors from parent reference frame to the child reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in parent reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Vectors mapped to the child reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "to_parent_position",
        .ml_meth = (void *)pyvl_reference_frame_to_parent_position,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "to_parent_position,(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map position vectors from child reference frame to the parent reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in child reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the parent reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "to_parent_velocity",
        .ml_meth = (void *)pyvl_reference_frame_to_parent_velocity,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "to_parent_velocity(position: array, velocity: array, time: float = 0.0, out_position: out_array | "
                  "None = None, out_velocity: out_array | None = None) -> tuple[out_array, out_array]\n"
                  "Map velocity vectors from local reference frame to the parent reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "position : (N, 3) array\n"
                  "    Array of :math:`N` position vectors in :math:`\\mathbb{R}^3` in local reference frame.\n"
                  "\n"
                  "velocity : (N, 3) array\n"
                  "    Array of :math:`N` velocity vectors in :math:`\\mathbb{R}^3` in local reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out_position : (N, 3) array, optional\n"
                  "    Array which receives the mapped position vectors. Must have the exact shape of ``position``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "out_velocity : (N, 3) array, optional\n"
                  "    Array which receives the mapped velocity vectors. Must have the exact shape of ``velocity``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the parent reference frame. If the ``out_position`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out_position`` was not specified,\n"
                  "    then a new array will be allocated."
                  "\n"
                  "(N, 3) array\n"
                  "    Velocity vectors mapped to the parent reference frame. If the ``out_velocity`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out_velocity`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "to_parent_vector",
        .ml_meth = (void *)pyvl_reference_frame_to_parent_vector,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "to_parent_vector(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map vectors from child reference frame from the parent reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in child reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Vectors mapped to the parent reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    // RF <-> Global
    {
        .ml_name = "from_global_position",
        .ml_meth = (void *)pyvl_reference_frame_from_global_position,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_global_position,(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map position vectors from global reference frame to the local reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in global reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the local reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "from_global_velocity",
        .ml_meth = (void *)pyvl_reference_frame_from_global_velocity,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_global_velocity(position: array, velocity: array, time: float = 0.0, out_position: out_array | "
                  "None = None, out_velocity: out_array | None = None) -> tuple[out_array, out_array]\n"
                  "Map velocity vectors from global reference frame to the local reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "position : (N, 3) array\n"
                  "    Array of :math:`N` position vectors in :math:`\\mathbb{R}^3` in global reference frame.\n"
                  "\n"
                  "velocity : (N, 3) array\n"
                  "    Array of :math:`N` velocity vectors in :math:`\\mathbb{R}^3` in global reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out_position : (N, 3) array, optional\n"
                  "    Array which receives the mapped position vectors. Must have the exact shape of ``position``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "out_velocity : (N, 3) array, optional\n"
                  "    Array which receives the mapped velocity vectors. Must have the exact shape of ``velocity``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the local reference frame. If the ``out_position`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out_position`` was not specified,\n"
                  "    then a new array will be allocated."
                  "\n"
                  "(N, 3) array\n"
                  "    Velocity vectors mapped to the local reference frame. If the ``out_velocity`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out_velocity`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "from_global_vector",
        .ml_meth = (void *)pyvl_reference_frame_from_global_vector,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "from_global_vector(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map vectors from global reference frame to the local reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in global reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Vectors mapped to the local reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "to_global_position",
        .ml_meth = (void *)pyvl_reference_frame_to_global_position,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "to_global_position,(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map position vectors from local reference frame to the global reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in local reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the global reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "to_global_velocity",
        .ml_meth = (void *)pyvl_reference_frame_to_global_velocity,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "to_global_velocity(position: array, velocity: array, time: float = 0.0, out_position: out_array | "
                  "None = None, out_velocity: out_array | None = None) -> tuple[out_array, out_array]\n"
                  "Map velocity vectors from local reference frame to the global reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "position : (N, 3) array\n"
                  "    Array of :math:`N` position vectors in :math:`\\mathbb{R}^3` in local reference frame.\n"
                  "\n"
                  "velocity : (N, 3) array\n"
                  "    Array of :math:`N` velocity vectors in :math:`\\mathbb{R}^3` in local reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out_position : (N, 3) array, optional\n"
                  "    Array which receives the mapped position vectors. Must have the exact shape of ``position``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "out_velocity : (N, 3) array, optional\n"
                  "    Array which receives the mapped velocity vectors. Must have the exact shape of ``velocity``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Position vectors mapped to the global reference frame. If the ``out_position`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out_position`` was not specified,\n"
                  "    then a new array will be allocated."
                  "\n"
                  "(N, 3) array\n"
                  "    Velocity vectors mapped to the global reference frame. If the ``out_velocity`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out_velocity`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    {
        .ml_name = "to_global_vector",
        .ml_meth = (void *)pyvl_reference_frame_to_global_vector,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "to_global_vector(x: array, time: float = 0.0, out: out_array | None = None) -> out_array\n"
                  "Map vectors from local reference frame from the global reference frame.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : (N, 3) array\n"
                  "    Array of :math:`N` vectors in :math:`\\mathbb{R}^3` in local reference frame.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    Time at which to evaluate the transformation.\n"
                  "\n"
                  "out : (N, 3) array, optional\n"
                  "    Array which receives the mapped vectors. Must have the exact shape of ``x``.\n"
                  "    It must also have the :class:`dtype` for :class:`numpy.double`, as well as be aligned,\n"
                  "    C-contiguous, and writable.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "(N, 3) array\n"
                  "    Vectors mapped to the global reference frame. If the ``out`` parameter was\n"
                  "    specified, this return value will be the same object. If ``out`` was not specified,\n"
                  "    then a new array will be allocated.",
    },
    // Adjusting + save/load
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
    // RF -> RF transformation
    {
        .ml_name = "transform_position",
        .ml_meth = (void *)pyvl_reference_frame_transform_position,
        .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "transform_position(x: array_like, src: ReferenceFrame | None = None, dst: ReferenceFrame | None = "
                  "None, time: float = 0.0, out: "
                  "out_array | None = None) -> out_array\n"
                  "Transform a position vector from one reference frame to another.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : array_like\n"
                  "    Position vectors to transform.\n"
                  "\n"
                  "start : ReferenceFrame, optional\n"
                  "    Reference frame the position vectors are given in. If not given, the global\n"
                  "    reference frame is assumed.\n"
                  "\n"
                  "end : ReferenceFrame, optional\n"
                  "    Reference frame the resulting vectors should be given in. If not given, the\n"
                  "    global reference frame is assumed.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    What time the transformations should be taken at. Only relevant if the\n"
                  "    reference frames have time-dependant motion.\n"
                  "\n"
                  "out : array, optional\n"
                  "    Output array to write the output to. If not given a new one is created.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "array\n"
                  "    Array of position vectors. If ``out`` was given, then the reference to it\n"
                  "    is returned.\n",
    },
    {
        .ml_name = "transform_velocity",
        .ml_meth = (void *)pyvl_reference_frame_transform_vel_pos,
        .ml_flags = METH_CLASS | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "transform_position(position: array_like, velocity: array_like, src: ReferenceFrame | None = None, "
                  "dst: ReferenceFrame | None = None, time: float = 0.0, out_position: out_array | None = None, "
                  "out_velocity: out_array | None = None) -> out_array\n"
                  "Transform position and velocity vectors from one reference frame to another.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "position : array_like\n"
                  "    Position vectors to transform.\n"
                  "\n"
                  "velocity : array_like\n"
                  "    Velocity vectors to transform.\n"
                  "\n"
                  "start : ReferenceFrame, optional\n"
                  "    Reference frame the position vectors are given in. If not given, the global\n"
                  "    reference frame is assumed.\n"
                  "\n"
                  "end : ReferenceFrame, optional\n"
                  "    Reference frame the resulting vectors should be given in. If not given, the\n"
                  "    global reference frame is assumed.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    What time the transformations should be taken at. Only relevant if the\n"
                  "    reference frames have time-dependant motion.\n"
                  "\n"
                  "out_position : array, optional\n"
                  "    Output array to write the output positions to. If not given a new one is\n"
                  "    created.\n"
                  "\n"
                  "out_velocity : array, optional\n"
                  "    Output array to write the output velocity to. If not given a new one is\n"
                  "    created.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "array\n"
                  "    Array of position vectors. If ``out_position`` was given, then the reference\n"
                  "    to it is returned.\n"
                  "\n"
                  "array\n"
                  "    Array of velocity vectors. If ``out_velocity`` was given, then the reference\n"
                  "    to it is returned.\n",
    },
    {
        .ml_name = "transform_vectors",
        .ml_meth = (void *)pyvl_reference_frame_transform_vector,
        .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "transform_vector(x: array_like, src: ReferenceFrame | None = None, dst: ReferenceFrame | None = "
                  "None, time: float = 0.0, out: "
                  "out_array | None = None) -> out_array\n"
                  "Transform a vector vector from one reference frame to another.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "x : array_like\n"
                  "    Vector vectors to transform.\n"
                  "\n"
                  "start : ReferenceFrame, optional\n"
                  "    Reference frame the vector vectors are given in. If not given, the global\n"
                  "    reference frame is assumed.\n"
                  "\n"
                  "end : ReferenceFrame, optional\n"
                  "    Reference frame the resulting vectors should be given in. If not given, the\n"
                  "    global reference frame is assumed.\n"
                  "\n"
                  "time : float, default: 0.0\n"
                  "    What time the transformations should be taken at. Only relevant if the\n"
                  "    reference frames have time-dependant motion.\n"
                  "\n"
                  "out : array, optional\n"
                  "    Output array to write the output to. If not given a new one is created.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "array\n"
                  "    Array of vector vectors. If ``out`` was given, then the reference to it\n"
                  "    is returned.\n",
    },
    //
    // Determining motion and ancestry
    {
        .ml_name = "moved_relative_to",
        .ml_meth = (void *)pyvl_reference_frame_moved_relative_to,
        .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
        .ml_doc = "moved_relative_to(other: ReferenceFrame | None, t_start: float, t_end: float, tol: float) -> bool\n"
                  "Check if the reference frame had motion relative to another.\n"
                  "\n"
                  "This function is intended to be used to determine if relative induction matrices\n"
                  "need to be recomputed.\n"
                  "\n"
                  "The motion is determined by computing the relative transformation between the two\n"
                  "reference frames at these two times. From there, two things are considered:\n"
                  "\n"
                  "- Does the difference in relative offset at the two times have the magnitude\n"
                  "  below ``tol``?\n"
                  "- Does the largest value of the relative orientation angles have the absolute\n"
                  "  value below ``tol``?\n"
                  "\n"
                  "If any of these criteria is met, the reference frames are considered to have\n"
                  "moved.\n"
                  "\n"
                  "Parameters\n"
                  "----------\n"
                  "other : ReferenceFrame | None\n"
                  "    Reference frame to compare it to. ``None`` corresponds to the global reference frame.\n"
                  "\n"
                  "t_start : float\n"
                  "    First time to compare to.\n"
                  "\n"
                  "t_end : float\n"
                  "    Second time to compare to.\n"
                  "\n"
                  "tol : float\n"
                  "    How much difference is allowed for the two reference frames to not\n"
                  "    be considered moving.\n"
                  "\n"
                  "Returns\n"
                  "-------\n"
                  "bool\n"
                  "    Indication if the two reference frames have moved with respect to one another.\n",
    },
    {
        .ml_name = "common_ancestor",
        .ml_meth = (void *)pyvl_reference_frame_common_ancestor,
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
        .ml_doc =
            "common_ancestor(other: ReferenceFrame | None) -> ReferenceFrame | None\n"
            "Find the first common ancestor with another reference frame.\n"
            "\n"
            "This function is intended to find the shortest transformation needed by the\n"
            "two reference frames.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "other : ReferenceFrame or None\n"
            "    The reference frame to find the ancestor with. ``None`` corresponds to the global reference frame.\n"
            "\n"
            "Returns\n"
            "-------\n"
            "ReferenceFrame of None\n"
            "    The nearest common ancestor of the two reference frames.\n",
    },
    {0},
};

PyDoc_STRVAR(pyvl_reference_frame_type_docstring,
             "ReferenceFrame(offset: VecLike3 | Callable = (0, 0, 0), theta: VecLike3 | Callable = (0, 0, 0), "
             "velocity: VecLike3 | Callable = (0, 0, 0), rotation: VecLike3 | Callable = (0, 0, 0), "
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
             "for a point at :math:`\\vec{r}_P` with velocity :math:`\\vec{v}_P` are given by\n"
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
             "velocity : VecLike3 or Callable, default: (0, 0, 0)\n"
             "    Linear velocity of the reference frame. Can be a constant vector or a callable\n"
             "    returning the velocity at time t.\n"
             "\n"
             "rotation : VecLike3 or Callable, default: (0, 0, 0)\n"
             "    Angular velocity (rotation rate) of the reference frame. Can be a constant vector\n"
             "    or a callable returning the rotation rate at time t.\n"
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
