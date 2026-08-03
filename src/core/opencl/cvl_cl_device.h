#pragma once
/*
 * OpenCL device discovery and selection.
 *
 * A plain value struct: discovery fills a cvl_cl_device_t with the raw
 * cl_device_id/cl_platform_id plus a cached snapshot of the device
 * info (fixed-size strings - no heap allocation).  No destroy is
 * needed; devices are stack/value objects.
 *
 * Example:
 * @code
 *   cvl_cl_device_t dev;
 *   cvl_cl_status_t st = cvl_cl_device_first_gpu(&dev);
 *   ...
 * @endcode
 */

#include "../common.h"
#include "cvl_cl_common.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Device selection descriptor                                        */
/* ------------------------------------------------------------------ */

typedef enum
{
    CVL_CL_PLATFORM_FILTER_NONE,  /**< No platform filtering (all platforms are candidates). */
    CVL_CL_PLATFORM_FILTER_INDEX, /**< Select a specific platform by index. */
    CVL_CL_PLATFORM_FILTER_NAME,  /**< Select platform whose name contains @ref platform_name_substring. */
} cvl_cl_platform_filter_type_t;

typedef struct
{
    cvl_cl_platform_filter_type_t type;
    union {
        unsigned platform_index;             /**< For CVL_CL_PLATFORM_FILTER_INDEX. */
        const char *platform_name_substring; /**< For CVL_CL_PLATFORM_FILTER_NAME (case-sensitive substring match). */
    };
} cvl_cl_platform_filter_t;

/* ------------------------------------------------------------------ */
/* Cached device info (filled once during discovery)                   */
/* ------------------------------------------------------------------ */

enum
{
    CVL_DEVICE_VERSION_MAX_LEN = 64, /**< Maximum length of the CL_DEVICE_VERSION string. */
    CVL_DEVICE_NAME_MAX_LEN = 128,   /**< Maximum length of the CL_DEVICE_NAME string. */
    CVL_DEVICE_VENDOR_MAX_LEN = 64,  /**< Maximum length of the CL_DEVICE_VENDOR string. */
    CVL_DEVICE_DRIVER_MAX_LEN = 64,  /**< Maximum length of the CL_DRIVER_VERSION string. */
};

typedef struct
{
    char name[CVL_DEVICE_NAME_MAX_LEN];             /**< CL_DEVICE_NAME. */
    char vendor[CVL_DEVICE_VENDOR_MAX_LEN];         /**< CL_DEVICE_VENDOR. */
    char version[CVL_DEVICE_VERSION_MAX_LEN];       /**< CL_DEVICE_VERSION string (e.g. "OpenCL 3.0"). */
    char driver_version[CVL_DEVICE_DRIVER_MAX_LEN]; /**< CL_DRIVER_VERSION. */
    size_t max_work_group_size;                     /**< CL_DEVICE_MAX_WORK_GROUP_SIZE. */
    size_t max_work_item_dims;                      /**< CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS (max 3). */
    size_t max_work_item_sizes[3];                  /**< CL_DEVICE_MAX_WORK_ITEM_SIZES. */
    cl_ulong local_mem_size;                        /**< CL_DEVICE_LOCAL_MEM_SIZE (bytes). */
    cl_ulong global_mem_size;                       /**< CL_DEVICE_GLOBAL_MEM_SIZE (bytes). */
    cl_ulong max_mem_alloc_size;                    /**< CL_DEVICE_MAX_MEM_ALLOC_SIZE (bytes). */
    cl_uint max_compute_units;                      /**< CL_DEVICE_MAX_COMPUTE_UNITS. */
    cl_uint address_bits;                           /**< CL_DEVICE_ADDRESS_BITS. */
    cl_bool available;                              /**< CL_DEVICE_AVAILABLE. */
    cl_bool compiler_available;                     /**< CL_DEVICE_COMPILER_AVAILABLE. */
    size_t preferred_wg_multiple;                   /**< 0 until queried via a specific kernel (filled lazily). */
} cvl_cl_device_info_t;

/* ------------------------------------------------------------------ */
/* Device handle (value type)                                          */
/* ------------------------------------------------------------------ */

struct cvl_cl_device_t
{
    cl_device_id id;
    cl_platform_id platform_id;
    unsigned platform_index; /**< Index in the platform list used during discovery. */
    cvl_cl_device_info_t info;
};

typedef struct cvl_cl_device_t cvl_cl_device_t;

/* ------------------------------------------------------------------ */
/* Discovery                                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Discover OpenCL devices matching a platform filter and device type.
 *
 * Enumerates all platforms (optionally narrowed by @p platform_filter)
 * and collects devices of the requested @p desired_types into
 * @p out_devices (up to @p max_devices entries).  @p out_count always
 * receives the total number of matches, even if it exceeds the array
 * capacity.
 *
 * @param platform_filter Platform filter (see cvl_cl_platform_filter_t).
 * @param max_devices     Capacity of @p out_devices.
 * @param out_devices     Array of @p max_devices device handles (value types).
 * @param desired_types   OR-ed cl_device_type bits (e.g. CL_DEVICE_TYPE_GPU).
 * @param out_count       Filled with the number of matching devices found.
 * @return CVL_CL_SUCCESS, or CVL_CL_ERR_DEVICE_NOT_FOUND if nothing matched.
 */
cvl_cl_status_t cvl_cl_device_discover(cvl_cl_platform_filter_t platform_filter, unsigned max_devices,
                                       cvl_cl_device_t out_devices[max_devices], const cl_device_type desired_types,
                                       unsigned *out_count);

/**
 * @brief Convenience wrapper around @ref cvl_cl_device_discover to find the first GPU device.
 *
 * @param out_device Pointer to a single device handle to fill.
 *
 * @return CVL_CL_SUCCESS on success, or an error code on failure.
 */
cvl_cl_status_t cvl_cl_device_first_gpu(cvl_cl_device_t *out_device);

/**
 * @brief Convenience wrapper around @ref cvl_cl_device_discover to find the first CPU device.
 *
 * @param out_device Pointer to a single device handle to fill.
 *
 * @return CVL_CL_SUCCESS on success, or an error code on failure.
 */
cvl_cl_status_t cvl_cl_device_first_cpu(cvl_cl_device_t *out_device);

/* ------------------------------------------------------------------ */
/* Backend identification                                             */
/* ------------------------------------------------------------------ */

/**
 * @brief Detect the Intel NEO CPU OpenCL backend.
 *
 * The NEO CPU backend (the experimental `libcpu_device.so` / `libOclCpuBackEnd.so`
 * path bundled with the Intel Graphics Compute Runtime) miscompiles kernels that
 * index `__local` memory with data-dependent indices loaded from global memory -
 * see `intel-neo-cpu-bug.md` at the repository root for the full report and the
 * minimal repro.
 *
 * Detection is a conservative heuristic:
 *   - device type is CPU,
 *   - vendor contains "intel",
 *   - CL_DEVICE_VERSION contains the NEO-CPU marker `(Build 0)`
 *     (the classic Intel CPU runtime reports e.g. "OpenCL 2.1 LINUX" and
 *     NEO GPU devices report e.g. "OpenCL 3.0 NEO").
 *
 * @param device Device handle (may be NULL).
 * @return true if the device is the Intel NEO CPU backend, false otherwise.
 */
bool cvl_cl_device_is_intel_neo_cpu(const cvl_cl_device_t *device);
