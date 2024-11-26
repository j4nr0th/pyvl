//
// Created by jan on 23.11.2024.
//

#ifndef MODULE_H
#define MODULE_H

#define PY_SSIZE_T_CLEAN

#ifdef __GNUC__
#define CDUST_INTERNAL __attribute__((visibility("hidden")))
#define CDUST_EXTERNAL __attribute__((visibility("default")))
#define CDUST_ARRAY_ARG(arr, sz) arr[sz]
#define CDUST_EXPECT_CONDITION(x) (__builtin_expect(x, 1))
#endif

#ifndef CDUST_EXPECT_CONDITION
#define CDUST_EXPECT_CONDITION(x) (x)
#endif

#ifndef CDUST_INTERNAL
#define CDUST_INTERNAL
#endif

#ifndef CDUST_EXTERNAL
#define CDUST_EXTERNAL
#endif

#ifndef CDUST_ARRAY_ARG
#define CDUST_ARRAY_ARG(arr, sz) *arr
#endif

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cdust
#endif

#include <Python.h>
#include <stdio.h>

#endif // MODULE_H
