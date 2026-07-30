#include "cvl_cl_common.h"
#include <CL/cl.h>

const char *cvl_cl_status_str(const cvl_cl_status_t status)
{
    switch (status)
    {
    case CVL_CL_SUCCESS:
        return "Success";
    case CVL_CL_ERR_PLATFORM:
        return "Platform query failed";
    case CVL_CL_ERR_DEVICE_NOT_FOUND:
        return "No matching OpenCL device found";
    case CVL_CL_ERR_DEVICE:
        return "Generic device error";
    case CVL_CL_ERR_INVALID_SELECTOR:
        return "Invalid device selection descriptor";
    case CVL_CL_ERR_CONTEXT:
        return "Context creation failed";
    case CVL_CL_ERR_QUEUE:
        return "Command-queue creation failed";
    case CVL_CL_ERR_PROGRAM:
        return "Program creation failed";
    case CVL_CL_ERR_PROGRAM_BUILD:
        return "Program build failed (see build log)";
    case CVL_CL_ERR_KERNEL:
        return "Kernel creation failed";
    case CVL_CL_ERR_KERNEL_ARG:
        return "Kernel argument setting failed";
    case CVL_CL_ERR_BUFFER:
        return "Buffer allocation failed";
    case CVL_CL_ERR_BUFFER_SIZE:
        return "Buffer size exceeds device limit";
    case CVL_CL_ERR_BUFFER_MAP:
        return "Buffer map/unmap failed";
    case CVL_CL_ERR_NDRANGE:
        return "Kernel execution (NDRange) failed";
    case CVL_CL_ERR_READ_WRITE:
        return "Buffer read/write failed";
    case CVL_CL_ERR_COPY:
        return "Buffer copy failed";
    case CVL_CL_ERR_EVENT:
        return "Event operation failed";
    case CVL_CL_ERR_FINISH:
        return "Queue finish/flush failed";
    case CVL_CL_ERR_BARRIER:
        return "Queue barrier failed";
    case CVL_CL_ERR_MEMORY:
        return "Host memory allocation failed";
    case CVL_CL_ERR_INVALID_PARAM:
        return "Invalid parameter (NULL pointer, bad size)";
    case CVL_CL_ERR_NOT_FOUND:
        return "Entity not found";
    case CVL_CL_ERR_INTERNAL:
        return "Internal / unexpected error";
    }
    return "Unknown status code";
}

cvl_cl_status_t cvl_cl_status_from_cl_int(const int err)
{
    switch (err)
    {
    case CL_SUCCESS:
        return CVL_CL_SUCCESS;
    case CL_DEVICE_NOT_FOUND:
        return CVL_CL_ERR_DEVICE_NOT_FOUND;
    case CL_DEVICE_NOT_AVAILABLE:
        return CVL_CL_ERR_DEVICE;
    case CL_COMPILER_NOT_AVAILABLE:
        return CVL_CL_ERR_PROGRAM;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return CVL_CL_ERR_BUFFER;
    case CL_OUT_OF_RESOURCES:
    case CL_OUT_OF_HOST_MEMORY:
        return CVL_CL_ERR_MEMORY;
    case CL_BUILD_PROGRAM_FAILURE:
        return CVL_CL_ERR_PROGRAM_BUILD;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return CVL_CL_ERR_INVALID_PARAM;
    case CL_INVALID_BUFFER_SIZE:
        return CVL_CL_ERR_BUFFER_SIZE;
    case CL_INVALID_VALUE:
    case CL_INVALID_DEVICE_TYPE:
    case CL_INVALID_PLATFORM:
    case CL_INVALID_DEVICE:
    case CL_INVALID_CONTEXT:
    case CL_INVALID_QUEUE_PROPERTIES:
    case CL_INVALID_COMMAND_QUEUE:
    case CL_INVALID_HOST_PTR:
    case CL_INVALID_MEM_OBJECT:
    case CL_INVALID_KERNEL_NAME:
    case CL_INVALID_KERNEL_DEFINITION:
    case CL_INVALID_KERNEL:
    case CL_INVALID_ARG_INDEX:
    case CL_INVALID_ARG_VALUE:
    case CL_INVALID_ARG_SIZE:
    case CL_INVALID_KERNEL_ARGS:
    case CL_INVALID_WORK_DIMENSION:
    case CL_INVALID_WORK_GROUP_SIZE:
    case CL_INVALID_WORK_ITEM_SIZE:
    case CL_INVALID_GLOBAL_OFFSET:
    case CL_INVALID_EVENT_WAIT_LIST:
    case CL_INVALID_EVENT:
    case CL_INVALID_OPERATION:
    case CL_INVALID_GL_OBJECT:
    case CL_INVALID_MIP_LEVEL:
    case CL_INVALID_GLOBAL_WORK_SIZE:
    case CL_INVALID_PROPERTY:
    case CL_INVALID_IMAGE_DESCRIPTOR:
    case CL_INVALID_IMAGE_SIZE:
    case CL_INVALID_SAMPLER:
    case CL_INVALID_BINARY:
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return CVL_CL_ERR_INVALID_PARAM;
    default:
        return CVL_CL_ERR_INTERNAL;
    }
}
