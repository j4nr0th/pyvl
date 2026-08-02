#include "cvl_cl_device.h"
#include "cvl_cl_helpers.h"

#include <ctype.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Platform query helper                                              */
/* ------------------------------------------------------------------ */

/**
 * @brief Maximum number of platforms we enumerate.  If the system has more,
 *        CVL_CL_ERR_PLATFORM is returned rather than silently truncating.
 */
enum
{
    CVL_CL_MAX_PLATFORMS = 64
};

/** @brief Maximum number of devices per platform. */
enum
{
    CVL_CL_MAX_DEVICES_PER_PLATFORM = 64
};

/* ------------------------------------------------------------------ */
/* Info query helper                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Query all device info fields and cache them in @p info.
 *
 * @param dev        OpenCL device ID.
 * @param info       Output info struct (cleared first).
 * @param allocator  Allocator for string properties (NULL = default).
 * @return CVL_CL_SUCCESS or error.
 */
static cvl_cl_status_t query_device_info(cl_device_id dev, cvl_cl_device_info_t *info, const allocator_t *allocator)
{
    cl_int err;

    /* Clear the struct first (preserving allocator). */
    const allocator_t *saved = info->allocator;
    memset(info, 0, sizeof(*info));
    info->allocator = saved;

    /* String properties - query size, allocate, then query. */
    struct
    {
        cl_device_info param;
        char **p_str;
    } strings[] = {
        {CL_DEVICE_NAME, &info->name},
        {CL_DEVICE_VENDOR, &info->vendor},
        {CL_DEVICE_VERSION, &info->version},
        {CL_DRIVER_VERSION, &info->driver_version},
    };
    for (size_t i = 0; i < sizeof(strings) / sizeof(strings[0]); ++i)
    {
        size_t sz = 0;
        err = clGetDeviceInfo(dev, strings[i].param, 0, NULL, &sz);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);
        *strings[i].p_str = (char *)cl_alloc(allocator, sz);
        if (!*strings[i].p_str)
            return CVL_CL_ERR_MEMORY;
        err = clGetDeviceInfo(dev, strings[i].param, sz, *strings[i].p_str, NULL);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);
    }

/* Scalar queries using a helper macro. */
#define QUERY(param, field)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        err = clGetDeviceInfo(dev, (param), sizeof(info->field), &info->field, NULL);                                  \
        if (err != CL_SUCCESS)                                                                                         \
            return cvl_cl_status_from_cl_int(err);                                                                     \
    } while (0)

    QUERY(CL_DEVICE_MAX_WORK_GROUP_SIZE, max_work_group_size);
    QUERY(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, max_work_item_dims);

    /* CL_DEVICE_MAX_WORK_ITEM_SIZES is an array - special handling. */
    {
        size_t query_dims = info->max_work_item_dims > 3 ? 3 : info->max_work_item_dims;
        /* We only cache the first 3 dims. */
        size_t raw[3] = {0, 0, 0};
        err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(raw), raw, NULL);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);
        for (size_t d = 0; d < query_dims && d < 3; ++d)
            info->max_work_item_sizes[d] = raw[d];
    }

    QUERY(CL_DEVICE_LOCAL_MEM_SIZE, local_mem_size);
    QUERY(CL_DEVICE_GLOBAL_MEM_SIZE, global_mem_size);
    QUERY(CL_DEVICE_MAX_MEM_ALLOC_SIZE, max_mem_alloc_size);
    QUERY(CL_DEVICE_MAX_COMPUTE_UNITS, max_compute_units);
    QUERY(CL_DEVICE_ADDRESS_BITS, address_bits);
    QUERY(CL_DEVICE_AVAILABLE, available);
    QUERY(CL_DEVICE_COMPILER_AVAILABLE, compiler_available);

#undef QUERY

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* destroy helper (free strings)                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Free all string fields in a device info struct.
 *
 * @param info  Device info whose strings were allocated via info->allocator.
 */
static void destroy_info(cvl_cl_device_info_t *info)
{
    const allocator_t *a = info->allocator;
    cl_free(a, info->name);
    cl_free(a, info->vendor);
    cl_free(a, info->version);
    cl_free(a, info->driver_version);
    info->name = NULL;
    info->vendor = NULL;
    info->version = NULL;
    info->driver_version = NULL;
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_device_discover(const cvl_cl_device_sel_t selectors[], unsigned max_devices, unsigned *out_count,
                                       cvl_cl_device_t out_devices[], const allocator_t *allocator)
{
    if (!selectors || !out_count || (!out_devices && max_devices > 0))
        return CVL_CL_ERR_INVALID_PARAM;

    const allocator_t *a = cl_resolve_allocator(allocator);
    *out_count = 0;

    /* --- Enumerate platforms --- */
    cl_uint n_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &n_platforms);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    if (n_platforms == 0)
        return CVL_CL_ERR_DEVICE_NOT_FOUND;
    if (n_platforms > CVL_CL_MAX_PLATFORMS)
        n_platforms = CVL_CL_MAX_PLATFORMS;

    cl_platform_id platforms[CVL_CL_MAX_PLATFORMS];
    err = clGetPlatformIDs(n_platforms, platforms, NULL);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    /* --- Evaluate selectors to pick platform + device type --- */
    cl_platform_id target_platform = NULL;
    cl_device_type target_type = CL_DEVICE_TYPE_ALL;

    for (const cvl_cl_device_sel_t *sel = selectors; sel->type != CVL_CL_DEVICE_SEL_NONE; ++sel)
    {
        switch (sel->type)
        {
        case CVL_CL_DEVICE_SEL_TYPE:
            target_type = sel->device_type;
            break;

        case CVL_CL_DEVICE_SEL_PLATFORM_INDEX:
            if (sel->platform_index < n_platforms)
                target_platform = platforms[sel->platform_index];
            break;

        case CVL_CL_DEVICE_SEL_PLATFORM_NAME: {
            /* Find the first platform whose name contains the substring. */
            for (cl_uint p = 0; p < n_platforms; ++p)
            {
                size_t sz = 0;
                err = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 0, NULL, &sz);
                if (err != CL_SUCCESS)
                    return cvl_cl_status_from_cl_int(err);
                char *name = (char *)cl_alloc(a, sz);
                if (!name)
                    return CVL_CL_ERR_MEMORY;
                err = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sz, name, NULL);
                if (err != CL_SUCCESS)
                {
                    cl_free(a, name);
                    return cvl_cl_status_from_cl_int(err);
                }
                if (strstr(name, sel->platform_name_substring) != NULL)
                {
                    target_platform = platforms[p];
                    cl_free(a, name);
                    break;
                }
                cl_free(a, name);
            }
            break;
        }

        default:
            return CVL_CL_ERR_INVALID_SELECTOR;
        }
    }

    /* --- Enumerate devices on the selected (or all) platform(s) --- */
    cl_uint n_platforms_to_search = (target_platform != NULL) ? 1 : n_platforms;
    cl_uint write_idx = 0;
    cvl_cl_status_t status = CVL_CL_SUCCESS;

    for (cl_uint p = 0; p < n_platforms_to_search; ++p)
    {
        const cl_uint pi = (target_platform != NULL) ? 0 : p;
        const cl_platform_id plat = (target_platform != NULL) ? target_platform : platforms[p];

        cl_uint n_devs = 0;
        err = clGetDeviceIDs(plat, target_type, 0, NULL, &n_devs);
        if (err != CL_SUCCESS)
        {
            /* CL_DEVICE_NOT_FOUND for this type on this platform - skip. */
            if (err == CL_DEVICE_NOT_FOUND)
                continue;
            return cvl_cl_status_from_cl_int(err);
        }
        if (n_devs > CVL_CL_MAX_DEVICES_PER_PLATFORM)
            n_devs = CVL_CL_MAX_DEVICES_PER_PLATFORM;
        if (n_devs == 0)
            continue;

        cl_device_id dev_ids[CVL_CL_MAX_DEVICES_PER_PLATFORM];
        err = clGetDeviceIDs(plat, target_type, n_devs, dev_ids, NULL);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);

        for (cl_uint d = 0; d < n_devs && write_idx < max_devices; ++d)
        {
            /* Check if device is available. */
            cl_bool available = CL_FALSE;
            err = clGetDeviceInfo(dev_ids[d], CL_DEVICE_AVAILABLE, sizeof(available), &available, NULL);
            if (err != CL_SUCCESS)
                continue;
            if (!available)
                continue;

            cvl_cl_device_t *dev = &out_devices[write_idx];
            dev->id = dev_ids[d];
            dev->platform_id = plat;
            dev->platform_index = pi;
            dev->info.allocator = a;

            status = query_device_info(dev_ids[d], &dev->info, a);
            if (status != CVL_CL_SUCCESS)
            {
                /* On failure, destroy what we have so far and return. */
                for (unsigned ci = 0; ci < write_idx; ++ci)
                    destroy_info(&out_devices[ci].info);
                return status;
            }

            write_idx++;
        }
    }

    *out_count = write_idx;
    if (write_idx == 0)
        return CVL_CL_ERR_DEVICE_NOT_FOUND;

    return CVL_CL_SUCCESS;
}

void cvl_cl_device_destroy(cvl_cl_device_t *device)
{
    if (!device)
        return;
    destroy_info(&device->info);
    device->id = NULL;
    device->platform_id = NULL;
}

/* ------------------------------------------------------------------ */
/* Backend identification                                              */
/* ------------------------------------------------------------------ */

/**
 * @brief Portable case-insensitive substring search (avoids strcasestr,
 *        which is not available on MSVC).
 */
static bool contains_ci(const char *haystack, const char *needle)
{
    if (!haystack || !needle)
        return false;

    const size_t nlen = strlen(needle);
    if (nlen == 0)
        return true;

    for (const char *p = haystack; *p; ++p)
    {
        size_t i = 0;
        while (i < nlen && p[i] && tolower((unsigned char)p[i]) == tolower((unsigned char)needle[i]))
            ++i;
        if (i == nlen)
            return true;
        if (!p[i])
            break;
    }
    return false;
}

bool cvl_cl_device_is_intel_neo_cpu(const cvl_cl_device_t *device)
{
    if (!device || !device->id)
        return false;

    /* Must be a CPU device. */
    cl_device_type type = 0;
    if (clGetDeviceInfo(device->id, CL_DEVICE_TYPE, sizeof type, &type, NULL) != CL_SUCCESS)
        return false;
    if ((type & CL_DEVICE_TYPE_CPU) == 0)
        return false;

    /* Vendor must be Intel. */
    if (!contains_ci(device->info.vendor, "intel"))
        return false;

    /* NEO CPU backend marker: "OpenCL 3.0 (Build 0)".
     * The classic Intel CPU runtime reports e.g. "OpenCL 2.1 LINUX" and
     * NEO GPU devices report e.g. "OpenCL 3.0 NEO". */
    if (!device->info.version || !strstr(device->info.version, "(Build 0)"))
        return false;

    return true;
}
