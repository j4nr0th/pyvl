#define PY_ARRAY_UNIQUE_SYMBOL cvl
#include "module.h"
#include "barneshuobject.h"
#include "geoidobject.h"
#include "meshobject.h"
#include "multipoleobject.h"
#include "referenceframeobject.h"
#include "transformationplaneobject.h"

#include "methods.h"
#include <numpy/arrayobject.h>

static int create_type_and_add_it(PyType_Spec *const specs, PyObject *const mod, PyTypeObject **const p_out)
{
    PyTypeObject *const type = (PyTypeObject *)PyType_FromModuleAndSpec(mod, specs, NULL);
    if (!type)
        return -1;

    const char *name = type->tp_name;
    // Search for the last dot in the name
    for (const char *pos = name; *pos; ++pos)
    {
        if (*pos == '.' && *(pos + 1))
            name = pos + 1;
    }

    const int res = PyModule_AddObjectRef(mod, name, (PyObject *)type);
    Py_DECREF(type);
    *p_out = type;

    return res;
}

/**
 * Create module types from their TypeSpecs and add them to the module.
 *
 * @param mod Module to add types to.
 * @return Non-zero on error
 */
static int cvl_module_add_types(PyObject *mod)
{
    // Get the state from the module.
    module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return -1;

    // We are gonna loop over these pairs, so just pack them into an anonymous struct
    const struct
    {
        PyType_Spec *spec;
        PyTypeObject **dst;
    } types_to_add[] = {
        {&pyvl_geoid_typespec, &state->geoid_type},
        {&pyvl_reference_frame_typespec, &state->rf_type},
        {&pyvl_transformation_plane_typespec, &state->transformation_plane_type},
        {&pyvl_mesh_typespec, &state->mesh_type},
        {&pyvl_multipole_typespec, &state->multipole_type},
        {&pyvl_bh_tree_typespec, &state->bh_tree_type},
        {0},
    };
    for (unsigned i = 0; types_to_add[i].spec != NULL; ++i)
    {
        const int res = create_type_and_add_it(types_to_add[i].spec, mod, types_to_add[i].dst);
        if (res != 0)
            return res;
    }

    const int res = PyModule_AddFunctions(mod, cvl_methods);
    if (res != 0)
        return res;

    // Just add the INVALID_ID at the end here
    return PyModule_AddIntConstant(mod, "INVALID_ID", INVALID_ID);
}

PyModuleDef cvl_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = PYVL_CMODULE_NAME,
    .m_doc = "The C implementation part of PyVL",
    .m_size = sizeof(module_state_t),
    .m_slots =
        (PyModuleDef_Slot[]){
            {Py_mod_exec, cvl_module_add_types}, // Function adds types and constants to the module
            {0},                                 // Sentinel
        },
};

PyMODINIT_FUNC PyInit_cvl(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
        return NULL;

    return PyModuleDef_Init(&cvl_module);

    // if (PyModule_AddType(mod, &pyvl_line_type))
    //     goto failed;
    // if (PyModule_AddType(mod, &pyvl_surface_type))
    //     goto failed;
    // if (PyModule_AddType(mod, &pyvl_mesh_type))
    //     goto failed;
    // if (PyModule_AddType(mod, &pyvl_reference_frame_type))
    //     goto failed;
}
