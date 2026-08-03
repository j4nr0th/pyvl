cmake_minimum_required(VERSION 3.29)

# OpenCL tests for the cvl_cl wrapper.
# These tests require a working OpenCL runtime; they are only built
# when OpenCL was found by the root CMakeLists.txt.

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(TEST_MATH_LIB m)
endif ()

# ------------------------------------------------------------------
# Helper macro to create an OpenCL test executable
# ------------------------------------------------------------------
macro(add_cvl_cl_test name source)
    add_executable(${name} ${source}
        "${CMAKE_SOURCE_DIR}/test/test_common.c"
        "${CMAKE_SOURCE_DIR}/test/test_common.h")
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/src/core/opencl")
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/src/core/cl_compute")
    target_link_libraries(${name} PRIVATE static_cvl ${TEST_MATH_LIB} OpenCL::OpenCL)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

# ------------------------------------------------------------------
# Tests (listed from simplest to most complex)
# ------------------------------------------------------------------

# Test 1: Device discovery
add_cvl_cl_test(test_cvl_cl_device test_cvl_cl_device.c)

# Test 2: Context + queue
add_cvl_cl_test(test_cvl_cl_context test_cvl_cl_context.c)

# Test 3: Program compilation
add_cvl_cl_test(test_cvl_cl_program test_cvl_cl_program.c)

# Test 4: Kernel arg setting
add_cvl_cl_test(test_cvl_cl_kernel test_cvl_cl_kernel.c)

# Test 5: Buffer management
add_cvl_cl_test(test_cvl_cl_buffer test_cvl_cl_buffer.c)

# Test 6: Full NDRange round-trip (write -> kernel -> read)
add_cvl_cl_test(test_cvl_cl_ndrange test_cvl_cl_ndrange.c)

# Test 7: Host-side C17 .cl.h math validation (no OpenCL runtime needed)
add_cvl_cl_test(test_cvl_cl_math_host test_cvl_cl_math_host.c)
target_include_directories(test_cvl_cl_math_host PRIVATE "${CMAKE_SOURCE_DIR}/src/core")
target_include_directories(test_cvl_cl_math_host PRIVATE "${CMAKE_SOURCE_DIR}/src/core/opencl")

# Test 8: OpenCL device-side .cl.h compilation + cross-validation
add_cvl_cl_test(test_cvl_cl_device_kernel_compile test_cvl_cl_device_kernel_compile.c)
target_include_directories(test_cvl_cl_device_kernel_compile PRIVATE "${CMAKE_SOURCE_DIR}/src/core")
target_include_directories(test_cvl_cl_device_kernel_compile PRIVATE "${CMAKE_SOURCE_DIR}/src/core/opencl")
target_compile_definitions(test_cvl_cl_device_kernel_compile PRIVATE
    CVL_CL_SOURCE_DIR="${CMAKE_SOURCE_DIR}/src/core/opencl")

# Test 9: End-to-end direct N-body GPU evaluation
add_cvl_cl_test(test_cvl_cl_direct_sum test_cvl_cl_direct_sum.c)

# Test 10: Uniform flat octree builder validation
add_cvl_cl_test(test_cvl_cl_flat_tree test_cvl_cl_flat_tree.c)

# Test 11: BH flat-tree GPU evaluation (order=0, compare with CPU direct sum)
add_cvl_cl_test(test_cvl_cl_bh_eval test_cvl_cl_bh_eval.c)

# Test 11b: GPU multipole coefficient kernels (P2M/M2M) + tree-code eval
add_cvl_cl_test(test_cvl_cl_bh_coeffs test_cvl_cl_bh_coeffs.c)
target_include_directories(test_cvl_cl_bh_coeffs PRIVATE "${CMAKE_SOURCE_DIR}/src/core")
target_include_directories(test_cvl_cl_bh_coeffs PRIVATE "${CMAKE_SOURCE_DIR}/src/core/opencl")

# Test 11c: Persistent tree handle (build/eval jobs, rebuild reuse)
add_cvl_cl_test(test_cvl_cl_tree test_cvl_cl_tree.c)
target_include_directories(test_cvl_cl_tree PRIVATE "${CMAKE_SOURCE_DIR}/src/core")
target_include_directories(test_cvl_cl_tree PRIVATE "${CMAKE_SOURCE_DIR}/src/core/opencl")

# Test 12: GPU tree build pipeline + BH evaluation end-to-end
add_cvl_cl_test(test_cvl_cl_gpu_build test_cvl_cl_gpu_build.c)
target_include_directories(test_cvl_cl_gpu_build PRIVATE "${CMAKE_SOURCE_DIR}/src/core")
target_include_directories(test_cvl_cl_gpu_build PRIVATE "${CMAKE_SOURCE_DIR}/src/core/opencl")

# Test 13: GPU FMM evaluation (L2P) end-to-end — compare with CPU fmm_tree_eval
add_cvl_cl_test(test_cvl_cl_fmm_eval test_cvl_cl_fmm_eval.c)
target_include_directories(test_cvl_cl_fmm_eval PRIVATE "${CMAKE_SOURCE_DIR}/src/core")
target_include_directories(test_cvl_cl_fmm_eval PRIVATE "${CMAKE_SOURCE_DIR}/src/core/opencl")

# Test 14: Intel NEO CPU workaround radix kernels + radix policy API
add_cvl_cl_test(test_cvl_cl_radix_workaround test_cvl_cl_radix_workaround.c)
target_include_directories(test_cvl_cl_radix_workaround PRIVATE "${CMAKE_SOURCE_DIR}/src/core")
target_include_directories(test_cvl_cl_radix_workaround PRIVATE "${CMAKE_SOURCE_DIR}/src/core/opencl")

# Test 14: Benchmark — performance + accuracy comparison
add_cvl_cl_test(test_cvl_cl_benchmark test_cvl_cl_benchmark.c)
target_include_directories(test_cvl_cl_benchmark PRIVATE "${CMAKE_SOURCE_DIR}/src/core")
target_include_directories(test_cvl_cl_benchmark PRIVATE "${CMAKE_SOURCE_DIR}/src/core/opencl")

# Test 15: Chain dependency chaining (write -> ndrange -> read, compaction,
# extra wait events)
add_cvl_cl_test(test_cvl_cl_chain test_cvl_cl_chain.c)

# Test 16: Compute backend pack selection + kernel registry
add_cvl_cl_test(test_cvl_cl_compute test_cvl_cl_compute.c)
