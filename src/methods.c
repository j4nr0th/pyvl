#include "methods.h"

#include <numpy/ndarrayobject.h>
// Must be below the NUMPY include
#include "core/flow_solver.h"
#include "transformationplaneobject.h"

#include <cpyutl.h>

static PyObject *quad_induction(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    double vortex_cutoff;
    double vortex_far_approximation;
    double vortex_smallest_size;
    PyObject *py_pos, *py_circ, *py_target;
    const PyVL_TransformationPlane *symmetry_plane = NULL;
    PyArrayObject *out_velocity = NULL;
    Py_ssize_t n_threads = 1;

    // Parse arguments
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_cutoff, .kwname = "vortex_cutoff"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_far_approximation, .kwname = "vortex_far_approximation"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_smallest_size, .kwname = "vortex_smallest_size"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_pos, .kwname = "quad_positions"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_circ, .kwname = "quad_circulations"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_target, .kwname = "target_positions"},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&symmetry_plane,
                 .kwname = "symmetry_plane",
                 .optional = true},
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&out_velocity,
                    .kwname = "out_velocity",
                    .optional = true,
                },
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (n_threads < 0)
    {
        PyErr_SetString(PyExc_TypeError, "n_threads must be non-negative");
        return NULL;
    }

    transformation_plane_t sym_plane;
    bool has_symmetry = false;
    if (symmetry_plane && !Py_IsNone((PyObject *)symmetry_plane))
    {
        if (!PyObject_TypeCheck((PyObject *)symmetry_plane, state->transformation_plane_type))
        {
            PyErr_SetString(PyExc_TypeError, "symmetry_plane must be a TransformationPlane object");
            return NULL;
        }
        if (!pyvl_transformation_plane_ensure_time_invariant(symmetry_plane))
            return NULL;
        sym_plane.normal = symmetry_plane->normal.value.constant;
        sym_plane.origin = symmetry_plane->origin.value.constant;
        has_symmetry = true;
    }

    // Convert the input array-likes to arrays
    PyArrayObject *arr_pos = NULL, *arr_circ = NULL, *arr_target = NULL;

    if ((arr_pos = (PyArrayObject *)PyArray_FROMANY(py_pos, NPY_DOUBLE, 3, 3,
                                                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL ||
        (arr_circ = (PyArrayObject *)PyArray_FROMANY(py_circ, NPY_DOUBLE, 1, 1,
                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL ||
        (arr_target = (PyArrayObject *)PyArray_FROMANY(py_target, NPY_DOUBLE, 2, 2,
                                                       NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL)
    {
        Py_XDECREF(arr_target);
        Py_XDECREF(arr_circ);
        Py_XDECREF(arr_pos);
        return NULL;
    }

    const size_t n_elements = PyArray_DIM(arr_circ, 0);
    // Check the sizes are correct
    if (check_input_array(arr_pos, 3, (npy_intp[3]){(npy_intp)n_elements, 4, 3}, NPY_DOUBLE, 0, "quad_positions") < 0 ||
        check_input_array(arr_target, 2, (npy_intp[3]){(npy_intp)0, 3}, NPY_DOUBLE, 0, "target_positions") < 0)
    {
        Py_DECREF(arr_target);
        Py_DECREF(arr_circ);
        Py_DECREF(arr_pos);
        return NULL;
    }

    const size_t n_targets = PyArray_DIM(arr_target, 0);

    // If the output is present, ensure it has the correct size, otherwise create it
    const npy_intp out_dims[2] = {(npy_intp)n_targets, 3};
    if (out_velocity && !Py_IsNone((PyObject *)out_velocity))
    {
        if (check_input_array(out_velocity, 2, out_dims, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "out_velocity") < 0)
        {
            Py_DECREF(arr_target);
            Py_DECREF(arr_circ);
            Py_DECREF(arr_pos);
            return NULL;
        }
        Py_INCREF(out_velocity);
    }
    else
    {
        out_velocity = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
        if (!out_velocity)
        {
            Py_DECREF(arr_target);
            Py_DECREF(arr_circ);
            Py_DECREF(arr_pos);
            return NULL;
        }
    }

    const real3_t *const positions = PyArray_DATA(arr_pos);
    const real3_t *const target = PyArray_DATA(arr_target);
    const real_t *const circulations = PyArray_DATA(arr_circ);

    real3_t *const velocity = PyArray_DATA(out_velocity);
    // Clear the output
    Py_BEGIN_ALLOW_THREADS;
    memset(velocity, 0, sizeof(*velocity) * n_targets);
    for (size_t i = 0; i < n_elements; ++i)
    {
        const real_t circulation = circulations[i];
        const real3_t *const element_pos = positions + 4 * i;
        real3_t pos_start = element_pos[3];
        for (unsigned i_end = 0; i_end < 4; ++i_end)
        {
            const real3_t pos_end = element_pos[i_end];

            real3_t direction = real3_sub(pos_end, pos_start);
            const real_t mag = real3_mag(direction);
            if (mag < vortex_smallest_size)
            {
                // Filament is too short
                continue;
            }
            direction.x /= mag;
            direction.y /= mag;
            direction.z /= mag;

#pragma omp parallel for default(none) shared(circulation, pos_start, pos_end, direction, vortex_cutoff,               \
                                                  vortex_far_approximation, n_targets, target, velocity)               \
    num_threads(n_threads) schedule(guided)
            for (unsigned j = 0; j < n_targets; ++j)
            {
                const real3_t ind = real3_mul1(compute_filament_induction(vortex_cutoff, vortex_far_approximation,
                                                                          pos_start, pos_end, direction, target[j]),
                                               circulation);

                velocity[j] = real3_add(velocity[j], ind);
            }

            if (has_symmetry)
            {
                // We need to apply symmetry
#pragma omp parallel for default(none) shared(circulation, pos_start, pos_end, direction, vortex_cutoff,               \
                                                  vortex_far_approximation, n_targets, target, velocity, sym_plane)    \
    num_threads(n_threads) schedule(guided)
                for (unsigned j = 0; j < n_targets; ++j)
                {
                    const real3_t ind =
                        real3_mul1(compute_filament_induction(
                                       vortex_cutoff, vortex_far_approximation, pos_start, pos_end, direction,
                                       transformation_plane_transform_position(&sym_plane, target[j])),
                                   circulation);

                    velocity[j] = real3_add(velocity[j], transformation_plane_transform_vector(&sym_plane, ind));
                }
            }

            pos_start = pos_end;
        }
    }
    Py_END_ALLOW_THREADS;
    Py_DECREF(arr_target);
    Py_DECREF(arr_circ);
    Py_DECREF(arr_pos);

    return (PyObject *)out_velocity;
}

PyDoc_STRVAR(quad_induction_docstring,
             "quad_induction(vortex_cutoff: float, vortex_far_approximation: float, vortex_smallest_size: float, "
             "quad_positions: numpy.typing.ArrayLike, quad_circulations: numpy.typing.ArrayLike, "
             "target_positions: numpy.typing.ArrayLike, out_velocity: numpy.typing.NDArray[numpy.double] | None = "
             "None, n_threads: int) -> numpy.typing.NDArray[numpy.double]:\n"
             "Compute the influence of quadrilateral circulation filaments at input positions.\n"
             "\n"
             "This is mainly used for computing the influence of the wake, which contains quads,\n"
             "which are considered separate (hence no mesh).\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "vortex_cutoff : float\n"
             "    Minimum normal distance before clamping velocity to zero.\n"
             "\n"
             "vortex_far_approximation : float\n"
             "    Limit for applying arctan far field approximation.\n"
             "\n"
             "vortex_smallest_size : float\n"
             "    Minimum line length below which execution is skipped.\n"
             "\n"
             "quad_positions : (M, 4, 3) array\n"
             "    Array of positions of the corners of the quadrilateral filaments. The first\n"
             "    dimension corresponds to the filaments, while the second dimension corresponds to\n"
             "    the corners of each filament.\n"
             "\n"
             "quad_circulations : (M,) array\n"
             "    Array of circulations for each quadrilateral filament.\n"
             "\n"
             "target_positions : (K, 3) array\n"
             "    Array of positions at which to compute the velocity influence.\n"
             "\n"
             "out_velocity : (K, 3) array, optional\n"
             "    Output array to write the computed velocities to. If not given, a new one is\n"
             "    created.\n"
             "\n"
             "n_threads : int, default: 1\n"
             "    Number of threads to use for computing the velocities.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "(K, 3) array\n"
             "    Array of velocity vectors at the target positions induced by the quadrilateral\n"
             "    filaments. If ``out_velocity`` was given, then the reference to it is returned.\n");

static PyObject *quad_normal_induction(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                       const PyObject *kwnames)
{
    const module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    double vortex_cutoff;
    double vortex_far_approximation;
    double vortex_smallest_size;
    PyObject *py_pos, *py_circ, *py_target, *py_normals;
    PyArrayObject *out_velocity = NULL;
    Py_ssize_t n_threads = 1;
    const PyVL_TransformationPlane *symmetry_plane = NULL;

    // Parse arguments
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_cutoff, .kwname = "vortex_cutoff"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_far_approximation, .kwname = "vortex_far_approximation"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_smallest_size, .kwname = "vortex_smallest_size"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_pos, .kwname = "quad_positions"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_circ, .kwname = "quad_circulations"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_target, .kwname = "target_positions"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_normals, .kwname = "target_normals"},
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&symmetry_plane,
                    .kwname = "symmetry_plane",
                    .optional = true,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&out_velocity,
                    .kwname = "out_velocity",
                    .optional = true,
                },
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (n_threads < 0)
    {
        PyErr_SetString(PyExc_TypeError, "n_threads must be non-negative");
        return NULL;
    }

    bool has_symmetry = false;
    transformation_plane_t sym_plane;
    if (symmetry_plane && !Py_IsNone((PyObject *)symmetry_plane))
    {
        if (!PyObject_TypeCheck((PyObject *)symmetry_plane, state->transformation_plane_type))
        {
            PyErr_SetString(PyExc_TypeError, "symmetry_plane must be a TransformationPlane object");
            return NULL;
        }
        if (!pyvl_transformation_plane_ensure_time_invariant(symmetry_plane))
            return NULL;
        sym_plane.normal = symmetry_plane->normal.value.constant;
        sym_plane.origin = symmetry_plane->origin.value.constant;
        has_symmetry = true;
    }

    // Convert the input array-likes to arrays
    PyArrayObject *arr_pos = NULL, *arr_circ = NULL, *arr_target = NULL, *arr_normals = NULL;

    if ((arr_pos = (PyArrayObject *)PyArray_FROMANY(py_pos, NPY_DOUBLE, 3, 3,
                                                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL ||
        (arr_circ = (PyArrayObject *)PyArray_FROMANY(py_circ, NPY_DOUBLE, 1, 1,
                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL ||
        (arr_target = (PyArrayObject *)PyArray_FROMANY(py_target, NPY_DOUBLE, 2, 2,
                                                       NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL ||
        (arr_normals = (PyArrayObject *)PyArray_FROMANY(py_normals, NPY_DOUBLE, 2, 2,
                                                        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL)
    {
        Py_XDECREF(arr_normals);
        Py_XDECREF(arr_target);
        Py_XDECREF(arr_circ);
        Py_XDECREF(arr_pos);
        return NULL;
    }

    const size_t n_elements = PyArray_DIM(arr_circ, 0);
    const size_t n_targets = PyArray_DIM(arr_target, 0);
    // Check the sizes are correct
    if (check_input_array(arr_pos, 3, (npy_intp[3]){(npy_intp)n_elements, 4, 3}, NPY_DOUBLE, 0, "quad_positions") < 0 ||
        check_input_array(arr_target, 2, (npy_intp[3]){(npy_intp)n_targets, 3}, NPY_DOUBLE, 0, "target_positions") <
            0 ||
        check_input_array(arr_normals, 2, (npy_intp[3]){(npy_intp)n_targets, 3}, NPY_DOUBLE, 0, "target_normals") < 0)
    {
        Py_DECREF(arr_normals);
        Py_DECREF(arr_target);
        Py_DECREF(arr_circ);
        Py_DECREF(arr_pos);
        return NULL;
    }

    // If the output is present, ensure it has the correct size, otherwise create it
    const npy_intp out_dims[1] = {(npy_intp)n_targets};
    if (out_velocity && !Py_IsNone((PyObject *)out_velocity))
    {
        if (check_input_array(out_velocity, 1, out_dims, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "out_velocity") < 0)
        {
            Py_DECREF(arr_normals);
            Py_DECREF(arr_target);
            Py_DECREF(arr_circ);
            Py_DECREF(arr_pos);
            return NULL;
        }
        Py_INCREF(out_velocity);
    }
    else
    {
        out_velocity = (PyArrayObject *)PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
        if (!out_velocity)
        {
            Py_DECREF(arr_normals);
            Py_DECREF(arr_target);
            Py_DECREF(arr_circ);
            Py_DECREF(arr_pos);
            return NULL;
        }
    }

    const real3_t *const positions = PyArray_DATA(arr_pos);
    const real3_t *const target = PyArray_DATA(arr_target);
    const real_t *const circulations = PyArray_DATA(arr_circ);
    const real3_t *const normals = PyArray_DATA(arr_normals);

    real_t *const velocity = PyArray_DATA(out_velocity);
    // Clear the output
    Py_BEGIN_ALLOW_THREADS;
    memset(velocity, 0, sizeof(*velocity) * n_targets);
    for (size_t i = 0; i < n_elements; ++i)
    {
        const real_t circulation = circulations[i];
        const real3_t *const element_pos = positions + 4 * i;
        real3_t pos_start = element_pos[3];
        for (unsigned i_end = 0; i_end < 4; ++i_end)
        {
            const real3_t pos_end = element_pos[i_end];

            real3_t direction = real3_sub(pos_end, pos_start);
            const real_t mag = real3_mag(direction);
            if (mag < vortex_smallest_size)
            {
                // Filament is too short
                continue;
            }
            direction.x /= mag;
            direction.y /= mag;
            direction.z /= mag;

#pragma omp parallel for default(none) shared(direction, circulation, pos_start, pos_end, vortex_cutoff,               \
                                                  vortex_far_approximation, n_targets, target, velocity, normals)      \
    num_threads(n_threads)
            for (unsigned j = 0; j < n_targets; ++j)
            {
                const real3_t ind = real3_mul1(compute_filament_induction(vortex_cutoff, vortex_far_approximation,
                                                                          pos_start, pos_end, direction, target[j]),
                                               circulation);
                const real_t normal_induction = real3_dot(ind, normals[j]);

                velocity[j] += normal_induction;
            }

            if (has_symmetry)
            {
                // We need to apply symmetry
#pragma omp parallel for default(none)                                                                                 \
    shared(direction, circulation, pos_start, pos_end, vortex_cutoff, vortex_far_approximation, n_targets, target,     \
               velocity, normals, sym_plane) num_threads(n_threads)
                for (unsigned j = 0; j < n_targets; ++j)
                {
                    const real3_t ind =
                        real3_mul1(compute_filament_induction(
                                       vortex_cutoff, vortex_far_approximation, pos_start, pos_end, direction,
                                       transformation_plane_transform_position(&sym_plane, target[j])),
                                   circulation);
                    const real_t normal_induction =
                        real3_dot(transformation_plane_transform_vector(&sym_plane, ind), normals[j]);

                    velocity[j] += normal_induction;
                }
            }

            pos_start = pos_end;
        }
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(arr_normals);
    Py_DECREF(arr_target);
    Py_DECREF(arr_circ);
    Py_DECREF(arr_pos);

    return (PyObject *)out_velocity;
}

PyDoc_STRVAR(quad_normal_induction_docstring,
             "def quad_normal_induction(vortex_cutoff: float, vortex_far_approximation: float, vortex_smallest_size: "
             "float, quad_positions: numpy.typing.ArrayLike, quad_circulations: "
             "numpy.typing.ArrayLike, "
             "target_positions: numpy.typingArrayLike, target_normals: numpy.typing.ArrayLike, out_velocity: "
             "numpy.typing.NDArray[numpy.double] | "
             "None = None, n_threads: int = 1) -> numpy.typing.NDArray[numpy.double]\n"
             "Compute the normal velocity induced by the quad filaments at the given positions.\n"
             "\n"
             "This is mainly used for computing the influence of the wake, which contains quads,\n"
             "which are considered separate (hence no mesh).\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "vortex_cutoff : float\n"
             "    Minimum normal distance before clamping velocity to zero.\n"
             "\n"
             "vortex_far_approximation : float\n"
             "    Limit for applying arctan far field approximation.\n"
             "\n"
             "vortex_smallest_size : float\n"
             "    Minimum line length below which execution is skipped.\n"
             "\n"
             "quad_positions : (M, 4, 3) array\n"
             "    Array of positions of the corners of the quadrilateral filaments. The first\n"
             "    dimension corresponds to the filaments, while the second dimension corresponds to\n"
             "    the corners of each filament.\n"
             "\n"
             "quad_circulations : (M,) array\n"
             "    Array of circulations for each quadrilateral filament.\n"
             "\n"
             "target_positions : (K, 3) array\n"
             "    Array of positions at which to compute the velocity influence.\n"
             "\n"
             "target_normals : (K, 3) array\n"
             "    Array of normal vectors at the target positions. The normal vectors should be\n"
             "    normalized.\n"
             "\n"
             "out_velocity : (K, 3) array, optional\n"
             "    Output array to write the computed velocities to. If not given, a new one is\n"
             "    created.\n"
             "\n"
             "n_threads : int, default: 1\n"
             "    Number of threads to use for computing the velocities.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "(K,) array\n"
             "    Array of normal components of velocity vectors at the target positions induced by\n"
             "    the quad filaments. If ``out_velocity`` was given, then the reference to it is\n"
             "    returned.\n");

static PyObject *line_induction(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    double vortex_cutoff;
    double vortex_far_approximation;
    double vortex_smallest_size;
    PyObject *py_pos, *py_circ, *py_target;
    const PyVL_TransformationPlane *symmetry_plane = NULL;
    PyArrayObject *out_velocity = NULL;
    Py_ssize_t n_threads = 1;

    // Parse arguments
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_cutoff, .kwname = "vortex_cutoff"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_far_approximation, .kwname = "vortex_far_approximation"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_smallest_size, .kwname = "vortex_smallest_size"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_pos, .kwname = "line_positions"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_circ, .kwname = "line_circulations"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_target, .kwname = "target_positions"},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = (void *)&symmetry_plane,
                 .kwname = "symmetry_plane",
                 .optional = true},
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&out_velocity,
                    .kwname = "out_velocity",
                    .optional = true,
                },
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (n_threads < 0)
    {
        PyErr_SetString(PyExc_TypeError, "n_threads must be non-negative");
        return NULL;
    }

    transformation_plane_t sym_plane;
    bool has_symmetry = false;
    if (symmetry_plane && !Py_IsNone((PyObject *)symmetry_plane))
    {
        if (!PyObject_TypeCheck((PyObject *)symmetry_plane, state->transformation_plane_type))
        {
            PyErr_SetString(PyExc_TypeError, "symmetry_plane must be a TransformationPlane object");
            return NULL;
        }
        if (!pyvl_transformation_plane_ensure_time_invariant(symmetry_plane))
            return NULL;
        sym_plane.normal = symmetry_plane->normal.value.constant;
        sym_plane.origin = symmetry_plane->origin.value.constant;
        has_symmetry = true;
    }

    // Convert the input array-likes to arrays
    PyArrayObject *arr_pos = NULL, *arr_circ = NULL, *arr_target = NULL;

    if ((arr_pos = (PyArrayObject *)PyArray_FROMANY(py_pos, NPY_DOUBLE, 3, 3,
                                                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL ||
        (arr_circ = (PyArrayObject *)PyArray_FROMANY(py_circ, NPY_DOUBLE, 1, 1,
                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL ||
        (arr_target = (PyArrayObject *)PyArray_FROMANY(py_target, NPY_DOUBLE, 2, 2,
                                                       NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL)
    {
        Py_XDECREF(arr_target);
        Py_XDECREF(arr_circ);
        Py_XDECREF(arr_pos);
        return NULL;
    }

    const size_t n_elements = PyArray_DIM(arr_circ, 0);
    // Check the sizes are correct
    if (check_input_array(arr_pos, 3, (npy_intp[3]){(npy_intp)n_elements, 2, 3}, NPY_DOUBLE, 0, "line_positions") < 0 ||
        check_input_array(arr_target, 2, (npy_intp[3]){(npy_intp)0, 3}, NPY_DOUBLE, 0, "target_positions") < 0)
    {
        Py_DECREF(arr_target);
        Py_DECREF(arr_circ);
        Py_DECREF(arr_pos);
        return NULL;
    }

    const size_t n_targets = PyArray_DIM(arr_target, 0);

    // If the output is present, ensure it has the correct size, otherwise create it
    const npy_intp out_dims[2] = {(npy_intp)n_targets, 3};
    if (out_velocity && !Py_IsNone((PyObject *)out_velocity))
    {
        if (check_input_array(out_velocity, 2, out_dims, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "out_velocity") < 0)
        {
            Py_DECREF(arr_target);
            Py_DECREF(arr_circ);
            Py_DECREF(arr_pos);
            return NULL;
        }
        Py_INCREF(out_velocity);
    }
    else
    {
        out_velocity = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
        if (!out_velocity)
        {
            Py_DECREF(arr_target);
            Py_DECREF(arr_circ);
            Py_DECREF(arr_pos);
            return NULL;
        }
    }

    const real3_t *const positions = PyArray_DATA(arr_pos);
    const real3_t *const target = PyArray_DATA(arr_target);
    const real_t *const circulations = PyArray_DATA(arr_circ);

    real3_t *const velocity = PyArray_DATA(out_velocity);
    // Clear the output
    Py_BEGIN_ALLOW_THREADS;
    memset(velocity, 0, sizeof(*velocity) * n_targets);
    for (size_t i = 0; i < n_elements; ++i)
    {
        const real_t circulation = circulations[i];
        const real3_t *const element_pos = positions + 2 * i;
        const real3_t pos_start = element_pos[0];
        const real3_t pos_end = element_pos[1];

        real3_t direction = real3_sub(pos_end, pos_start);
        const real_t mag = real3_mag(direction);
        if (mag < vortex_smallest_size)
        {
            // Filament is too short
            continue;
        }
        direction.x /= mag;
        direction.y /= mag;
        direction.z /= mag;

#pragma omp parallel for default(none) shared(circulation, pos_start, pos_end, direction, vortex_cutoff,               \
                                                  vortex_far_approximation, n_targets, target, velocity)               \
    num_threads(n_threads) schedule(guided)
        for (unsigned j = 0; j < n_targets; ++j)
        {
            const real3_t ind = real3_mul1(compute_filament_induction(vortex_cutoff, vortex_far_approximation,
                                                                      pos_start, pos_end, direction, target[j]),
                                           circulation);

            velocity[j] = real3_add(velocity[j], ind);
        }

        if (has_symmetry)
        {
            // We need to apply symmetry
#pragma omp parallel for default(none) shared(circulation, pos_start, pos_end, direction, vortex_cutoff,               \
                                                  vortex_far_approximation, n_targets, target, velocity, sym_plane)    \
    num_threads(n_threads) schedule(guided)
            for (unsigned j = 0; j < n_targets; ++j)
            {
                const real3_t ind = real3_mul1(
                    compute_filament_induction(vortex_cutoff, vortex_far_approximation, pos_start, pos_end, direction,
                                               transformation_plane_transform_position(&sym_plane, target[j])),
                    circulation);

                velocity[j] = real3_add(velocity[j], transformation_plane_transform_vector(&sym_plane, ind));
            }
        }
    }
    Py_END_ALLOW_THREADS;
    Py_DECREF(arr_target);
    Py_DECREF(arr_circ);
    Py_DECREF(arr_pos);

    return (PyObject *)out_velocity;
}

PyDoc_STRVAR(line_induction_docstring,
             "line_induction(vortex_cutoff: float, vortex_far_approximation: float, vortex_smallest_size: float, "
             "line_positions: numpy.typing.ArrayLike, line_circulations: numpy.typing.ArrayLike, "
             "target_positions: numpy.typing.ArrayLike, out_velocity: numpy.typing.NDArray[numpy.double] | None = "
             "None, n_threads: int) -> numpy.typing.NDArray[numpy.double]:\n"
             "Compute the influence of line circulation filaments at input positions.\n"
             "\n"
             "This is mainly used for computing the influence of the wake, which contains lines,\n"
             "which are considered separate (hence no mesh).\n"
             "\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "vortex_cutoff : float\n"
             "    Minimum normal distance before clamping velocity to zero.\n"
             "\n"
             "vortex_far_approximation : float\n"
             "    Limit for applying arctan far field approximation.\n"
             "\n"
             "vortex_smallest_size : float\n"
             "    Minimum line length below which execution is skipped.\n"
             "\n"
             "\n"
             "line_positions : (M, 2, 3) array\n"
             "    Array of positions of the endpoints of the line filaments. The first\n"
             "    dimension corresponds to the filaments, while the second dimension corresponds to\n"
             "    the endpoints of each filament.\n"
             "\n"
             "\n"
             "line_circulations : (M,) array\n"
             "    Array of circulations for each line filament.\n"
             "\n"
             "\n"
             "target_positions : (K, 3) array\n"
             "    Array of positions at which to compute the velocity influence.\n"
             "\n"
             "\n"
             "symmetry_plane : TransformationPlane, optional\n"
             "    If given, the influence of the line filaments is computed as if they were\n"
             "    mirrored across the given plane.\n"
             "\n"
             "\n"
             "out_velocity : (K, 3) array, optional\n"
             "    Output array to write the computed velocities to. If not given, a new one is\n"
             "    created.\n"
             "\n"
             "\n"
             "n_threads : int, default: 1\n"
             "    Number of threads to use for computing the velocities.\n"
             "\n"
             "\n"
             "Returns\n"
             "-------\n"
             "(K, 3) array\n"
             "    Array of velocity vectors at the target positions induced by the line\n"
             "    filaments. If ``out_velocity`` was given, then the reference to it is returned.\n");

static PyObject *line_normal_induction(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs,
                                       const PyObject *kwnames)
{
    const module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    double vortex_cutoff;
    double vortex_far_approximation;
    double vortex_smallest_size;
    PyObject *py_pos, *py_circ, *py_target, *py_normals;
    PyArrayObject *out_velocity = NULL;
    Py_ssize_t n_threads = 1;
    const PyVL_TransformationPlane *symmetry_plane = NULL;

    // Parse arguments
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_cutoff, .kwname = "vortex_cutoff"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_far_approximation, .kwname = "vortex_far_approximation"},
                {.type = CPYARG_TYPE_DOUBLE, .p_val = &vortex_smallest_size, .kwname = "vortex_smallest_size"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_pos, .kwname = "line_positions"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_circ, .kwname = "line_circulations"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_target, .kwname = "target_positions"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = (void *)&py_normals, .kwname = "target_normals"},
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&symmetry_plane,
                    .kwname = "symmetry_plane",
                    .optional = true,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = (void *)&out_velocity,
                    .kwname = "out_velocity",
                    .optional = true,
                },
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = true},
                {0},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (n_threads < 0)
    {
        PyErr_SetString(PyExc_TypeError, "n_threads must be non-negative");
        return NULL;
    }

    bool has_symmetry = false;
    transformation_plane_t sym_plane;
    if (symmetry_plane && !Py_IsNone((PyObject *)symmetry_plane))
    {
        if (!PyObject_TypeCheck((PyObject *)symmetry_plane, state->transformation_plane_type))
        {
            PyErr_SetString(PyExc_TypeError, "symmetry_plane must be a TransformationPlane object");
            return NULL;
        }
        if (!pyvl_transformation_plane_ensure_time_invariant(symmetry_plane))
            return NULL;
        sym_plane.normal = symmetry_plane->normal.value.constant;
        sym_plane.origin = symmetry_plane->origin.value.constant;
        has_symmetry = true;
    }

    // Convert the input array-likes to arrays
    PyArrayObject *arr_pos = NULL, *arr_circ = NULL, *arr_target = NULL, *arr_normals = NULL;

    if ((arr_pos = (PyArrayObject *)PyArray_FROMANY(py_pos, NPY_DOUBLE, 3, 3,
                                                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL ||
        (arr_circ = (PyArrayObject *)PyArray_FROMANY(py_circ, NPY_DOUBLE, 1, 1,
                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL ||
        (arr_target = (PyArrayObject *)PyArray_FROMANY(py_target, NPY_DOUBLE, 2, 2,
                                                       NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL ||
        (arr_normals = (PyArrayObject *)PyArray_FROMANY(py_normals, NPY_DOUBLE, 2, 2,
                                                        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) == NULL)
    {
        Py_XDECREF(arr_normals);
        Py_XDECREF(arr_target);
        Py_XDECREF(arr_circ);
        Py_XDECREF(arr_pos);
        return NULL;
    }

    const size_t n_elements = PyArray_DIM(arr_circ, 0);
    const size_t n_targets = PyArray_DIM(arr_target, 0);
    // Check the sizes are correct
    if (check_input_array(arr_pos, 3, (npy_intp[3]){(npy_intp)n_elements, 2, 3}, NPY_DOUBLE, 0, "line_positions") < 0 ||
        check_input_array(arr_target, 2, (npy_intp[3]){(npy_intp)n_targets, 3}, NPY_DOUBLE, 0, "target_positions") <
            0 ||
        check_input_array(arr_normals, 2, (npy_intp[3]){(npy_intp)n_targets, 3}, NPY_DOUBLE, 0, "target_normals") < 0)
    {
        Py_DECREF(arr_normals);
        Py_DECREF(arr_target);
        Py_DECREF(arr_circ);
        Py_DECREF(arr_pos);
        return NULL;
    }

    // If the output is present, ensure it has the correct size, otherwise create it
    const npy_intp out_dims[1] = {(npy_intp)n_targets};
    if (out_velocity && !Py_IsNone((PyObject *)out_velocity))
    {
        if (check_input_array(out_velocity, 1, out_dims, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, "out_velocity") < 0)
        {
            Py_DECREF(arr_normals);
            Py_DECREF(arr_target);
            Py_DECREF(arr_circ);
            Py_DECREF(arr_pos);
            return NULL;
        }
        Py_INCREF(out_velocity);
    }
    else
    {
        out_velocity = (PyArrayObject *)PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
        if (!out_velocity)
        {
            Py_DECREF(arr_normals);
            Py_DECREF(arr_target);
            Py_DECREF(arr_circ);
            Py_DECREF(arr_pos);
            return NULL;
        }
    }

    const real3_t *const positions = PyArray_DATA(arr_pos);
    const real3_t *const target = PyArray_DATA(arr_target);
    const real_t *const circulations = PyArray_DATA(arr_circ);
    const real3_t *const normals = PyArray_DATA(arr_normals);

    real_t *const velocity = PyArray_DATA(out_velocity);
    // Clear the output
    Py_BEGIN_ALLOW_THREADS;
    memset(velocity, 0, sizeof(*velocity) * n_targets);
    for (size_t i = 0; i < n_elements; ++i)
    {
        const real_t circulation = circulations[i];
        const real3_t *const element_pos = positions + 2 * i;
        const real3_t pos_start = element_pos[0];
        const real3_t pos_end = element_pos[1];

        real3_t direction = real3_sub(pos_end, pos_start);
        const real_t mag = real3_mag(direction);
        if (mag < vortex_smallest_size)
        {
            // Filament is too short
            continue;
        }
        direction.x /= mag;
        direction.y /= mag;
        direction.z /= mag;

#pragma omp parallel for default(none) shared(direction, circulation, pos_start, pos_end, vortex_cutoff,               \
                                                  vortex_far_approximation, n_targets, target, velocity, normals)      \
    num_threads(n_threads)
        for (unsigned j = 0; j < n_targets; ++j)
        {
            const real3_t ind = real3_mul1(compute_filament_induction(vortex_cutoff, vortex_far_approximation,
                                                                      pos_start, pos_end, direction, target[j]),
                                           circulation);
            const real_t normal_induction = real3_dot(ind, normals[j]);

            velocity[j] += normal_induction;
        }

        if (has_symmetry)
        {
            // We need to apply symmetry
#pragma omp parallel for default(none)                                                                                 \
    shared(direction, circulation, pos_start, pos_end, vortex_cutoff, vortex_far_approximation, n_targets, target,     \
               velocity, normals, sym_plane) num_threads(n_threads)
            for (unsigned j = 0; j < n_targets; ++j)
            {
                const real3_t ind = real3_mul1(
                    compute_filament_induction(vortex_cutoff, vortex_far_approximation, pos_start, pos_end, direction,
                                               transformation_plane_transform_position(&sym_plane, target[j])),
                    circulation);
                const real_t normal_induction =
                    real3_dot(transformation_plane_transform_vector(&sym_plane, ind), normals[j]);

                velocity[j] += normal_induction;
            }
        }
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(arr_normals);
    Py_DECREF(arr_target);
    Py_DECREF(arr_circ);
    Py_DECREF(arr_pos);

    return (PyObject *)out_velocity;
}

PyDoc_STRVAR(line_normal_induction_docstring,
             "def line_normal_induction(vortex_cutoff: float, vortex_far_approximation: float, vortex_smallest_size: "
             "float, line_positions: numpy.typing.ArrayLike, line_circulations: "
             "numpy.typing.ArrayLike, "
             "target_positions: numpy.typingArrayLike, target_normals: numpy.typing.ArrayLike, out_velocity: "
             "numpy.typing.NDArray[numpy.double] | "
             "None = None, n_threads: int = 1) -> numpy.typing.NDArray[numpy.double]\n"
             "Compute the normal velocity induced by the line filaments at the given positions.\n"
             "\n"
             "This is mainly used for computing the influence of the wake, which contains lines,\n"
             "which are considered separate (hence no mesh).\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "vortex_cutoff : float\n"
             "    Minimum normal distance before clamping velocity to zero.\n"
             "\n"
             "vortex_far_approximation : float\n"
             "    Limit for applying arctan far field approximation.\n"
             "\n"
             "vortex_smallest_size : float\n"
             "    Minimum line length below which execution is skipped.\n"
             "\n"
             "line_positions : (M, 2, 3) array\n"
             "    Array of positions of the endpoints of the line filaments. The first\n"
             "    dimension corresponds to the filaments, while the second dimension corresponds to\n"
             "    the endpoints of each filament.\n"
             "\n"
             "line_circulations : (M,) array\n"
             "    Array of circulations for each line filament.\n"
             "\n"
             "target_positions : (K, 3) array\n"
             "    Array of positions at which to compute the velocity influence.\n"
             "\n"
             "target_normals : (K, 3) array\n"
             "    Array of normal vectors at the target positions. The normal vectors should be\n"
             "    normalized.\n"
             "\n"
             "symmetry_plane : TransformationPlane, optional\n"
             "    If given, the influence of the line filaments is computed as if they were\n"
             "    mirrored across the given plane.\n"
             "\n"
             "out_velocity : (K, 3) array, optional\n"
             "    Output array to write the computed velocities to. If not given, a new one is\n"
             "    created.\n"
             "\n"
             "n_threads : int, default: 1\n"
             "    Number of threads to use for computing the velocities.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "(K,) array\n"
             "    Array of normal components of velocity vectors at the target positions induced by\n"
             "    the line filaments. If ``out_velocity`` was given, then the reference to it is\n"
             "    returned.\n");

CVL_INTERNAL
PyMethodDef cvl_methods[] = {
    {
        .ml_name = "quad_induction",
        .ml_meth = (void *)quad_induction,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = quad_induction_docstring,
    },
    {
        .ml_name = "quad_normal_induction",
        .ml_meth = (void *)quad_normal_induction,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = quad_normal_induction_docstring,
    },
    {
        .ml_name = "line_induction",
        .ml_meth = (void *)line_induction,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = line_induction_docstring,
    },
    {
        .ml_name = "line_normal_induction",
        .ml_meth = (void *)line_normal_induction,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = line_normal_induction_docstring,
    },
    {0}, // Sentinel
};
