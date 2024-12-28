#define PY_ARRAY_UNIQUE_SYMBOL cvl
#include "module.h"
#include "geoidobject.h"
#include "lineobject.h"
#include "meshobject.h"
#include "referenceframeobject.h"
#include "surfaceobject.h"

#include <numpy/arrayobject.h>

static PyModuleDef cvl_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "cvl",
    .m_doc = "The C implementation part of PyVL",
    .m_size = -1,
    .m_methods = NULL,
    .m_slots = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL,
};

PyMODINIT_FUNC PyInit_cvl(void)
{
    // TODO: as a low priority, maybe add Perf maps for each subfile.
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
        return NULL;

    PyObject *const mod = PyModule_Create(&cvl_module);
    if (!mod)
        goto failed;

    if (PyModule_AddType(mod, &pyvl_geoid_type))
        goto failed;
    if (PyModule_AddType(mod, &pyvl_line_type))
        goto failed;
    if (PyModule_AddType(mod, &pyvl_surface_type))
        goto failed;
    if (PyModule_AddType(mod, &pyvl_mesh_type))
        goto failed;
    if (PyModule_AddType(mod, &pyvl_reference_frame_type))
        goto failed;
    if (PyModule_AddIntConstant(mod, "INVALID_ID", INVALID_ID))
        goto failed;

    return mod;
failed:
    Py_XDECREF(mod);
    return NULL;
}
