#pragma once
/*
 * OpenCL device discovery and selection.
 *
 * Design follows the cpyutl typed-descriptor pattern: the caller builds
 * a NULL-terminated array of @ref cvl_cl_device_sel_t descriptors using
 * designated initializers, and @ref cvl_cl_device_discover picks the
 * best matching device.
 *
 * Example:
 * @code
 *   cvl_cl_device_t dev;
 *   unsigned count = 0;
 *   cvl_cl_status_t st = cvl_cl_device_discover(
 *       (cvl_cl_device_sel_t[]){
 *           {.type = CVL_CL_DEVICE_SEL_TYPE, .device_type = CL_DEVICE_TYPE_GPU},
 *           {.type = CVL_CL_DEVICE_SEL_PLATFORM_NAME, .platform_name_substring = "NVIDIA"},
 *           {},
 *       },
 *       1, &count, &dev);
 * @endcode
 */

#include "cvl_cl_common.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Device selection descriptor                                        */
/* ------------------------------------------------------------------ */

typedef enum
{
    CVL_CL_DEVICE_SEL_NONE,           /**< Terminator — marks end of the selection array. */
    CVL_CL_DEVICE_SEL_TYPE,           /**< Select by CL_DEVICE_TYPE (GPU, CPU, etc.). */
    CVL_CL_DEVICE_SEL_PLATFORM_INDEX, /**< Select a specific platform by index. */
    CVL_CL_DEVICE_SEL_PLATFORM_NAME,  /**< Select platform whose name contains @ref platform_name_substring. */
} cvl_cl_device_sel_type_t;

typedef struct
{
    cvl_cl_device_sel_type_t type;
    union {
        cl_device_type device_type; /**< For CVL_CL_DEVICE_SEL_TYPE. */
        unsigned platform_index;    /**< For CVL_CL_DEVICE_SEL_PLATFORM_INDEX. */
        const char
            *platform_name_substring; /**< For CVL_CL_DEVICE_SEL_PLATFORM_NAME (case-sensitive substring match). */
    };
} cvl_cl_device_sel_t;

/* ------------------------------------------------------------------ */
/* Cached device info (filled once during discovery)                   */
/* ------------------------------------------------------------------ */

typedef struct
{
    char *name;                    /**< CL_DEVICE_NAME. */
    char *vendor;                  /**< CL_DEVICE_VENDOR. */
    char *version;                 /**< CL_DEVICE_VERSION string (e.g. "OpenCL 3.0"). */
    char *driver_version;          /**< CL_DRIVER_VERSION. */
    size_t max_work_group_size;    /**< CL_DEVICE_MAX_WORK_GROUP_SIZE. */
    size_t max_work_item_dims;     /**< CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS (max 3). */
    size_t max_work_item_sizes[3]; /**< CL_DEVICE_MAX_WORK_ITEM_SIZES. */
    cl_ulong local_mem_size;       /**< CL_DEVICE_LOCAL_MEM_SIZE (bytes). */
    cl_ulong global_mem_size;      /**< CL_DEVICE_GLOBAL_MEM_SIZE (bytes). */
    cl_ulong max_mem_alloc_size;   /**< CL_DEVICE_MAX_MEM_ALLOC_SIZE (bytes). */
    cl_uint max_compute_units;     /**< CL_DEVICE_MAX_COMPUTE_UNITS. */
    cl_uint address_bits;          /**< CL_DEVICE_ADDRESS_BITS. */
    cl_bool available;             /**< CL_DEVICE_AVAILABLE. */
    cl_bool compiler_available;    /**< CL_DEVICE_COMPILER_AVAILABLE. */
    size_t preferred_wg_multiple;  /**< 0 until queried via a specific kernel (filled lazily). */
} cvl_cl_device_info_t;

/* ------------------------------------------------------------------ */
/* Device handle (opaque)                                              */
/* ------------------------------------------------------------------ */

struct cvl_cl_device_t
{
    cl_device_id id;
    cl_platform_id platform_id;
    unsigned platform_index; /**< Index in the platform list used during discovery. */
    cvl_cl_device_info_t info;
};

/* ------------------------------------------------------------------ */
/* Discovery                                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Discover and select an OpenCL device.
 *
 * Enumerates all platforms and their devices, applying the given
 * selection criteria in order.  Each criterion narrows the candidate
 * set.  After all criteria are processed, the first remaining device
 * is selected.
 *
 * Typical usage with a single selection criterion:
 * @code
 *   cvl_cl_device_t dev;
 *   unsigned count = 0;
 *   cvl_cl_device_discover(
 *       (cvl_cl_device_sel_t[]){
 *           {.type = CVL_CL_DEVICE_SEL_TYPE, .device_type = CL_DEVICE_TYPE_GPU},
 *           {},
 *       },
 *       1, &count, &dev);
 *   if (count == 0) ... // no suitable device
 * @endcode
 *
 * @param selectors   NULL-terminated array of selection descriptors.
 * @param max_devices Capacity of @p out_devices (pass 1 for a single device).
 * @param out_count   Filled with the number of matching devices found (may exceed max_devices).
 * @param out_devices Array of @p max_devices device handles.  Each handle
 *                    must be destroyed via @ref cvl_cl_device_destroy.
 * @return CVL_CL_SUCCESS on success, or an error code on failure.
 *         CVL_CL_ERR_DEVICE_NOT_FOUND if selectors matched nothing.
 */
cvl_cl_status_t cvl_cl_device_discover(const cvl_cl_device_sel_t selectors[], unsigned max_devices, unsigned *out_count,
                                       cvl_cl_device_t out_devices[]);

/**
 * @brief Destroy a device handle, freeing cached info strings.
 *
 * Does NOT call clReleaseDevice (the device handle remains valid for
 * the lifetime of the context).  Frees host-side allocations.
 *
 * @param device Device handle to destroy (may be NULL).
 */
void cvl_cl_device_destroy(cvl_cl_device_t *device);

/* ------------------------------------------------------------------ */
/* Accessors                                                          */
/* ------------------------------------------------------------------ */

/** @brief Return the raw cl_device_id. */
static inline cl_device_id cvl_cl_device_id(const cvl_cl_device_t *device)
{
    return device ? device->id : NULL;
}

/** @brief Return the platform_id associated with this device. */
static inline cl_platform_id cvl_cl_device_platform_id(const cvl_cl_device_t *device)
{
    return device ? device->platform_id : NULL;
}

/** @brief Return the cached device info struct. */
static inline const cvl_cl_device_info_t *cvl_cl_device_info(const cvl_cl_device_t *device)
{
    return device ? &device->info : NULL;
}
