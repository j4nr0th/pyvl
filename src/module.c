#define PY_ARRAY_UNIQUE_SYMBOL cdust
#include "module.h"
#include "geoidobject.h"
#include "lineobject.h"
#include "meshobject.h"
#include "referenceframeobject.h"
#include "surfaceobject.h"

#include <numpy/arrayobject.h>

static PyModuleDef cdust_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "cdust",
    .m_doc = "The C implementation part of PyDUST",
    .m_size = -1,
    .m_methods = nullptr,
    .m_slots = nullptr,
    .m_traverse = nullptr,
    .m_clear = nullptr,
    .m_free = nullptr,
};

PyMODINIT_FUNC PyInit_cdust(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
        return nullptr;

    PyObject *const mod = PyModule_Create(&cdust_module);
    if (!mod)
        goto failed;

    if (PyModule_AddType(mod, &pydust_geoid_type))
        goto failed;
    if (PyModule_AddType(mod, &pydust_line_type))
        goto failed;
    if (PyModule_AddType(mod, &pydust_surface_type))
        goto failed;
    if (PyModule_AddType(mod, &pydust_mesh_type))
        goto failed;
    if (PyModule_AddType(mod, &pydust_reference_frame_type))
        goto failed;

    return mod;
failed:
    Py_XDECREF(mod);
    return nullptr;
}
