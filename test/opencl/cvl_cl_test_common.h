#pragma once
/*
 * Shared helpers for the OpenCL tests (test/opencl/).
 *
 * The CVL_CL_CHECK macro was removed from the library (cvl_cl_common.h)
 * as part of the internal-API cleanup - the library never used it.  It
 * lives here now, for tests only.
 */

#include "cvl_cl.h"

/**
 * @brief Call an OpenCL wrapper function, jumping to @p label on failure.
 *
 * Assigns the result to the in-scope `status` variable (tests declare
 * `cvl_cl_status_t status = CVL_CL_SUCCESS;`) and prints the failing
 * expression + status string for diagnostics.
 */
#define CVL_CL_CHECK(stmt, label)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        status = (stmt);                                                                                               \
        if (status != CVL_CL_SUCCESS)                                                                                  \
        {                                                                                                              \
            fprintf(stderr, "CVL_CL_CHECK failed at %s:%d: %s -> %s\n", __FILE__, __LINE__, #stmt,                     \
                    cvl_cl_status_str(status));                                                                        \
            goto label;                                                                                                \
        }                                                                                                              \
    } while (0)
