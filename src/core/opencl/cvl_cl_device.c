#include "cvl_cl_device.h"

#include <assert.h>
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
 * @param dev   OpenCL device ID.
 * @param info  Output info struct (cleared first).
 * @return CVL_CL_SUCCESS or error.
 */
static cvl_cl_status_t query_device_info(cl_device_id dev, cvl_cl_device_info_t *info)
{
    cl_int err;

    /* Clear the struct first. */
    memset(info, 0, sizeof(*info));

    /* String properties - query size, allocate, then query. */
    struct
    {
        cl_device_info param;
        char *p_str;
        size_t max_len;
    } strings[] = {
        {CL_DEVICE_NAME, info->name, CVL_DEVICE_NAME_MAX_LEN},
        {CL_DEVICE_VENDOR, info->vendor, CVL_DEVICE_VENDOR_MAX_LEN},
        {CL_DEVICE_VERSION, info->version, CVL_DEVICE_VERSION_MAX_LEN},
        {CL_DRIVER_VERSION, info->driver_version, CVL_DEVICE_DRIVER_MAX_LEN},
    };
    for (size_t i = 0; i < sizeof(strings) / sizeof(strings[0]); ++i)
    {
        size_t sz = 0;
        err = clGetDeviceInfo(dev, strings[i].param, strings[i].max_len, strings[i].p_str, &sz);
        if (sz > strings[i].max_len)
        {
            // Buffer might be too small, so try with a bigger static buffer, then truncate to our output buffer size.
            enum
            {
                TMP_BUF_SIZE = 512
            };
            if (sz > TMP_BUF_SIZE)
                // Too bad, we tried
                return cvl_cl_status_from_cl_int(err);

            char tmp[TMP_BUF_SIZE];
            err = clGetDeviceInfo(dev, strings[i].param, TMP_BUF_SIZE, tmp, NULL);
            if (err != CL_SUCCESS)
                return cvl_cl_status_from_cl_int(err);
            memcpy(strings[i].p_str, tmp, strings[i].max_len - 1);
            strings[i].p_str[strings[i].max_len - 1] = '\0';
        }
        else if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);
    }

    // Helper macro
#define QUERY_FIELD(param, field) (err = clGetDeviceInfo(dev, (param), sizeof(info->field), &info->field, NULL))

    if (QUERY_FIELD(CL_DEVICE_MAX_WORK_GROUP_SIZE, max_work_group_size) != CL_SUCCESS ||
        QUERY_FIELD(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, max_work_item_dims) != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

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

    if (QUERY_FIELD(CL_DEVICE_LOCAL_MEM_SIZE, local_mem_size) != CL_SUCCESS ||
        QUERY_FIELD(CL_DEVICE_GLOBAL_MEM_SIZE, global_mem_size) != CL_SUCCESS ||
        QUERY_FIELD(CL_DEVICE_MAX_MEM_ALLOC_SIZE, max_mem_alloc_size) != CL_SUCCESS ||
        QUERY_FIELD(CL_DEVICE_MAX_COMPUTE_UNITS, max_compute_units) != CL_SUCCESS ||
        QUERY_FIELD(CL_DEVICE_ADDRESS_BITS, address_bits) != CL_SUCCESS ||
        QUERY_FIELD(CL_DEVICE_AVAILABLE, available) != CL_SUCCESS ||
        QUERY_FIELD(CL_DEVICE_COMPILER_AVAILABLE, compiler_available) != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

#undef QUERY_FIELD

    return CVL_CL_SUCCESS;
}

/**
 * @brief Check the platform for devices of desired types and fill the output array.
 *
 * @param pi            Platform index.
 * @param plat          Platform ID.
 * @param target_types  Desired device types (bitfield).
 * @param max_devices   Maximum number of devices to fill in out_devices.
 * @param out_devices   Output array of devices (must have capacity for max_devices).
 * @param out_count     Filled with the number of devices found (may exceed max_devices).
 *
 * @return CVL_CL_SUCCESS on success, or an error code on failure.
 */
static cvl_cl_status_t check_platform_for_devices(const unsigned pi, const cl_platform_id plat,
                                                  const cl_device_type target_types, const unsigned max_devices,
                                                  cvl_cl_device_t out_devices[max_devices], unsigned *out_count)
{

    cl_uint n_devs = 0;
    cl_int err = clGetDeviceIDs(plat, target_types, 0, NULL, &n_devs);
    if (err != CL_SUCCESS)
    {
        /* CL_DEVICE_NOT_FOUND for this type on this platform - skip. */
        if (err == CL_DEVICE_NOT_FOUND)
            return CVL_CL_SUCCESS;
        return cvl_cl_status_from_cl_int(err);
    }
    if (n_devs > CVL_CL_MAX_DEVICES_PER_PLATFORM)
        n_devs = CVL_CL_MAX_DEVICES_PER_PLATFORM;
    if (n_devs == 0)
        return CVL_CL_SUCCESS;

    cl_device_id dev_ids[CVL_CL_MAX_DEVICES_PER_PLATFORM];
    err = clGetDeviceIDs(plat, target_types, n_devs, dev_ids, NULL);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    /* Count every available match; write at most max_devices entries. */
    unsigned written = 0;
    unsigned matches = 0;
    for (cl_uint d = 0; d < n_devs; ++d)
    {
        /* Check if device is available. */
        cl_bool available = CL_FALSE;
        err = clGetDeviceInfo(dev_ids[d], CL_DEVICE_AVAILABLE, sizeof(available), &available, NULL);
        if (err != CL_SUCCESS)
            continue;
        if (!available)
            continue;

        matches += 1;
        if (written >= max_devices)
            continue;

        cvl_cl_device_t *dev = out_devices + written;
        dev->id = dev_ids[d];
        dev->platform_id = plat;
        dev->platform_index = pi;

        const cvl_cl_status_t status = query_device_info(dev_ids[d], &dev->info);
        if (status != CVL_CL_SUCCESS)
            return status;

        written += 1;
    }
    *out_count = matches;
    return CVL_CL_SUCCESS;
}

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

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_device_discover(const cvl_cl_platform_filter_t platform_filter, unsigned max_devices,
                                       cvl_cl_device_t out_devices[max_devices],
                                       const cl_device_type desired_device_types, unsigned *out_count)
{
    /* Internal module: NULL output pointers are contract violations. */
    assert(out_count != NULL);
    assert(out_devices != NULL || max_devices == 0);

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

    /* --- Evaluate the platform filter to pick candidate platform(s) --- */
    cl_uint p = n_platforms;

    switch (platform_filter.type)
    {
    case CVL_CL_PLATFORM_FILTER_NAME:
        /* Find the first platform whose name contains the substring. */
        for (p = 0; p < n_platforms; ++p)
        {
            enum
            {
                PLATFORM_NAME_MAX_LEN = 128,
            };
            // Contain the name substring in this scope
            {
                size_t sz = 0;
                char name[PLATFORM_NAME_MAX_LEN];
                err = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, PLATFORM_NAME_MAX_LEN, name, &sz);
                if (err != CL_SUCCESS)
                    return cvl_cl_status_from_cl_int(err);

                if (!contains_ci(name, platform_filter.platform_name_substring))
                {
                    continue;
                }
            }
            break;
        }
        if (p == n_platforms)
            return CVL_CL_ERR_NOT_FOUND; // No platform matched the substring

        break;

    case CVL_CL_PLATFORM_FILTER_INDEX:
        p = platform_filter.platform_index;
        if (p >= n_platforms)
            return CVL_CL_ERR_NOT_FOUND; // Index out of range.
        break;

    case CVL_CL_PLATFORM_FILTER_NONE:
        break;

    default:
        return CVL_CL_ERR_INVALID_SELECTOR;
    }

    if (p < n_platforms)
    {
        /* A single platform was selected: search it for desired device(s). */
        unsigned n_devs = 0;
        const cvl_cl_status_t status =
            check_platform_for_devices(p, platforms[p], desired_device_types, max_devices, out_devices, &n_devs);
        if (status != CVL_CL_SUCCESS)
            return status;
        *out_count = n_devs;
        if (n_devs == 0)
            return CVL_CL_ERR_DEVICE_NOT_FOUND;
        return CVL_CL_SUCCESS;
    }

    /* No filter: enumerate devices across all platforms, filling the output
     * array up to max_devices while still reporting the total match count. */
    unsigned written = 0;
    unsigned total = 0;
    cvl_cl_status_t status = CVL_CL_SUCCESS;

    for (cl_uint p = 0; p < n_platforms; ++p)
    {
        const cl_uint pi = p;
        const cl_platform_id plat = platforms[p];

        unsigned n_devs = 0;
        const unsigned capacity = (written < max_devices) ? (max_devices - written) : 0;
        status = check_platform_for_devices(pi, plat, desired_device_types, capacity, out_devices + written, &n_devs);
        if (status != CVL_CL_SUCCESS)
            return status;

        total += n_devs;
        written += (n_devs < capacity) ? n_devs : capacity;
    }

    *out_count = total;
    if (total == 0)
        return CVL_CL_ERR_DEVICE_NOT_FOUND;

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_device_first_gpu(cvl_cl_device_t *out_device)
{
    unsigned count = 0;
    cvl_cl_status_t status =
        cvl_cl_device_discover((cvl_cl_platform_filter_t){0}, 1, out_device, CL_DEVICE_TYPE_GPU, &count);
    if (status != CVL_CL_SUCCESS || count == 0)
        return CVL_CL_ERR_DEVICE_NOT_FOUND;

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_device_first_cpu(cvl_cl_device_t *out_device)
{
    unsigned count = 0;
    cvl_cl_status_t status =
        cvl_cl_device_discover((cvl_cl_platform_filter_t){0}, 1, out_device, CL_DEVICE_TYPE_CPU, &count);
    if (status != CVL_CL_SUCCESS || count == 0)
        return CVL_CL_ERR_DEVICE_NOT_FOUND;

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Backend identification                                              */
/* ------------------------------------------------------------------ */

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
    if (!strstr(device->info.version, "(Build 0)"))
        return false;

    return true;
}
