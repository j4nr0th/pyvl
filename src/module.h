//
// Created by jan on 23.11.2024.
//

#ifndef MODULE_H
#define MODULE_H

#define PY_SSIZE_T_CLEAN

#ifdef __GNUC__
#define CVL_INTERNAL __attribute__((visibility("hidden")))
#define CVL_EXTERNAL __attribute__((visibility("default")))
#define CVL_ARRAY_ARG(arr, sz) arr[sz]
#define CVL_EXPECT_CONDITION(x) (__builtin_expect(x, 1))
#endif

#ifndef CVL_EXPECT_CONDITION
#define CVL_EXPECT_CONDITION(x) (x)
#endif

#ifndef CVL_INTERNAL
#define CVL_INTERNAL
#endif

#ifndef CVL_EXTERNAL
#define CVL_EXTERNAL
#endif

#ifndef CVL_ARRAY_ARG
#define CVL_ARRAY_ARG(arr, sz) *arr
#endif

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cvl
#endif

#include <Python.h>
#include <stdio.h>

#endif // MODULE_H
