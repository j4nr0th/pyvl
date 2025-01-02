//
// Created by jan on 23.11.2024.
//

#ifndef MODULE_H
#define MODULE_H

#define PY_SSIZE_T_CLEAN

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cvl
#endif

#include "core/common.h"
#include <Python.h>
#include <stdio.h>

#endif // MODULE_H
