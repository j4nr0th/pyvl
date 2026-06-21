/**
 * Shared helper for time-dependent 3-vector fields.
 */
#pragma once

#include "core/common.h"
#include "module.h"

typedef enum
{
    PYVL_VEC_CONSTANT,
    PYVL_VEC_CALLABLE,
} pyvl_vec_value_type_t;

typedef struct
{
    pyvl_vec_value_type_t type;
    union {
        real3_t constant;
        PyObject *callable;
    } value;
} pyvl_vec_time_dependent_t;

/**
 * Initialize a time-dependent 3-vector field.
 *
 * @param field The field to initialize.
 * @param value The initial value, either a constant 3-vector or a callable.
 * @param field_name The name of the field, used for error messages.
 * @return true on success, false on failure.
 */
CVL_INTERNAL
bool pyvl_vec_time_dependent_init(pyvl_vec_time_dependent_t *field, PyObject *value, const char *field_name);

/**
 * Evaluate a time-dependent 3-vector field at a given time.
 *
 * @param field The field to evaluate.
 * @param time_arg The time at which to evaluate the field.
 * @param value The output value.
 * @return true on success, false on failure.
 */
CVL_INTERNAL
bool pyvl_vec_time_dependent_eval_c(const pyvl_vec_time_dependent_t *field, PyObject *time_arg, real3_t *value);

/**
 * Evaluate a time-dependent 3-vector field at a given time and return a Python object.
 *
 * @param field The field to evaluate.
 * @param time_arg The time at which to evaluate the field.
 * @return A Python object representing the value, or NULL on failure.
 */
CVL_INTERNAL
PyObject *pyvl_vec_time_dependent_eval_py(const pyvl_vec_time_dependent_t *field, PyObject *time_arg);

/**
 * Clear and deallocate a time-dependent 3-vector field.
 *
 * @param field The field to clear.
 */
CVL_INTERNAL
void pyvl_vec_time_dependent_clear(pyvl_vec_time_dependent_t *field);

/**
 * Serialize a time-dependent 3-vector field.
 *
 * @param entry The field to serialize.
 * @param key The key under which to store the serialized data.
 * @param hmap The Python HierarchicalMap to store the serialized data.
 * @param serializer The Python callable to use for serialization.
 * @return true on success, false on failure.
 */
CVL_INTERNAL
bool pyvl_vec_time_dependent_serialize(const pyvl_vec_time_dependent_t *entry, const char *key, PyObject *hmap,
                                       PyObject *serializer);

/**
 * Deserialize a time-dependent 3-vector field.
 *
 * @param key The key under which the serialized data is stored.
 * @param hmap The Python HierarchicalMap containing the serialized data.
 * @param deserializer The Python callable to use for deserialization.
 * @param entry The field to populate with the deserialized data.
 * @return true on success, false on failure.
 */
CVL_INTERNAL
bool pyvl_vec_time_dependent_deserialize(const char *key, PyObject *hmap, PyObject *deserializer,
                                         pyvl_vec_time_dependent_t *entry);
