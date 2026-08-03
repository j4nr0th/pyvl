#pragma once
/*
 * cvl_cl.h - umbrella header for the pyvl OpenCL wrapper.
 *
 * Includes all cvl_cl_* modules.  Functionality available only when
 * CVL_OPENCL is defined (set by CMake when OpenCL is found).
 *
 * Modules:
 *   cvl_cl_common  - status codes, precision, cross-compilation macros
 *   cvl_cl_device  - device discovery and selection
 *   cvl_cl_ctx     - context and command queue creation
 *   cvl_cl_program - program compilation
 *   cvl_cl_kernel  - kernel + typed argument descriptors
 *   cvl_cl_buffer  - capacity-tracked buffers
 *   cvl_cl_command - ndrange, transfers, events
 *   cvl_cl_chain   - dependency chaining for multi-stage pipelines
 */

#include "cvl_cl_buffer.h"
#include "cvl_cl_chain.h"
#include "cvl_cl_command.h"
#include "cvl_cl_common.h"
#include "cvl_cl_ctx.h"
#include "cvl_cl_device.h"
#include "cvl_cl_kernel.h"
#include "cvl_cl_program.h"
