cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "cvl")
endif ()

project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

set(CMAKE_C_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(OpenCL QUIET)
# set(Python_FIND_VIRTUALENV ONLY) # Needed for UV
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

# Find the NumPy headers
execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(
        APPEND CVL_MODULE_SOURCE_FILES
        src/module.c
        src/allocator.c
        src/geoidobject.c
        src/meshobject.c
        src/multipoleobject.c
        src/time_dependent_vector.c
        src/referenceframeobject.c
        src/transformationplaneobject.c
        src/barneshuobject.c
        src/fmmtreeobject.c
        src/methods.c
)

list(
        APPEND CVL_MODULE_HEADER_FILES
        src/module.h
        src/allocator.h
        src/geoidobject.h
        src/meshobject.h
        src/multipoleobject.h
        src/time_dependent_vector.h
        src/referenceframeobject.h
        src/transformationplaneobject.h
        src/barneshuobject.h
        src/fmmtreeobject.h
        src/methods.h
)

list(
        APPEND CVL_CORE_SOURCE_FILES
        src/core/solver_state.c
        src/core/common.c
        src/core/mesh.c
        src/core/mesh_io.c
        src/core/flow_solver.c
        src/core/transformation.c
        src/core/approx_atan.c
        src/core/multipole.c
        src/core/octree.c
        src/core/barnes_hut_tree.c
        src/core/fmm_operators.c
        src/core/fmm_tree.c
)

list(
        APPEND CVL_CORE_HEADER_FILES
        src/core/solver_state.h
        src/core/common.h
        src/core/mesh.h
        src/core/mesh_io.h
        src/core/flow_solver.h
        src/core/transformation.h
        src/core/approx_atan.h
        src/core/octree.h
        src/core/barnes_hut_tree.h
        src/core/cost_model.h
        src/core/fmm_operators.h
        src/core/fmm_tree.h
        src/core/cvl_radix_sort.h
)

# ---- OpenCL (optional) ----
if (OpenCL_FOUND)
    add_definitions(-DCVL_OPENCL)

    # Generated header embedding the kernel-source packs (see
    # gen_kernel_sources.py).  Placed in the build dir so the source
    # tree stays clean; the include path is added to static_cvl and cvl.
    set(CVL_CL_GEN_HEADER "${CMAKE_CURRENT_BINARY_DIR}/cvl_cl_kernel_sources.h")
    add_custom_command(
            OUTPUT "${CVL_CL_GEN_HEADER}"
            COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/cl_compute/gen_kernel_sources.py"
            --source-dir "${CMAKE_CURRENT_SOURCE_DIR}/src/core/opencl"
            --output "${CVL_CL_GEN_HEADER}"
            DEPENDS
            src/core/opencl/bh_build.cl.h
            src/core/opencl/bh_flat_eval.cl.h
            src/core/opencl/fmm_l2p.cl.h
            src/core/opencl/direct_sum.cl.h
            src/core/opencl/cvl_cl_types.h.cl
            src/core/opencl/cvl_cl_math.h.cl
            src/core/opencl/cvl_cl_multipole.h.cl
            src/core/opencl/cvl_cl_multipole_ops.h.cl
            src/core/opencl/cvl_cl_fmm_ops.h.cl
            src/core/cl_compute/gen_kernel_sources.py
            COMMENT "Generating cvl_cl_kernel_sources.h"
            VERBATIM
    )

    list(
            APPEND CVL_CORE_SOURCE_FILES
            src/core/opencl/cvl_cl_common.c
            src/core/opencl/cvl_cl_device.c
            src/core/opencl/cvl_cl_ctx.c
            src/core/opencl/cvl_cl_program.c
            src/core/opencl/cvl_cl_kernel.c
            src/core/opencl/cvl_cl_buffer.c
            src/core/opencl/cvl_cl_command.c
            src/core/opencl/cvl_cl_chain.c
    )

    list(
            APPEND CVL_CORE_SOURCE_FILES
            src/core/cl_compute/cvl_cl_staging_buffer.c
            src/core/cl_compute/cvl_cl_compute.c
            src/core/cl_compute/cvl_cl_flat_tree.c
            src/core/cl_compute/cvl_cl_gpu_tree_build.c
            src/core/cl_compute/cvl_cl_fmm_eval.c
    )

    list(
            APPEND CVL_CORE_HEADER_FILES
            src/core/opencl/cvl_cl.h
            src/core/opencl/cvl_cl_common.h
            src/core/opencl/cvl_cl_device.h
            src/core/opencl/cvl_cl_ctx.h
            src/core/opencl/cvl_cl_program.h
            src/core/opencl/cvl_cl_kernel.h
            src/core/opencl/cvl_cl_buffer.h
            src/core/opencl/cvl_cl_command.h
            src/core/opencl/cvl_cl_chain.h
            src/core/opencl/cvl_cl_helpers.h
    )

    list(
            APPEND CVL_CORE_HEADER_FILES
            src/core/cl_compute/cvl_cl_staging_buffer.h
            src/core/cl_compute/cvl_cl_compute.h
            src/core/cl_compute/cvl_cl_flat_tree.h
            src/core/cl_compute/cvl_cl_gpu_tree_build.h
            src/core/cl_compute/cvl_cl_fmm_eval.h
            "${CVL_CL_GEN_HEADER}"
    )
endif ()

add_library(static_cvl
        ${CVL_CORE_SOURCE_FILES}
        ${CVL_CORE_HEADER_FILES}
        src/core/multipole.c
        src/core/fmm_operators.c
        src/core/fmm_tree.c
)
set_property(TARGET static_cvl PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
target_include_directories(static_cvl PRIVATE src/core/opencl)
if (OpenCL_FOUND)
    target_include_directories(static_cvl PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
endif ()
target_link_libraries(static_cvl PUBLIC OpenMP::OpenMP_C)
if (OpenCL_FOUND)
    target_link_libraries(static_cvl PUBLIC OpenCL::OpenCL)
endif ()


Python_add_library(cvl MODULE ${CVL_CORE_SOURCE_FILES} ${CVL_CORE_HEADER_FILES} ${CVL_MODULE_SOURCE_FILES}
        ${CVL_MODULE_HEADER_FILES} WITH_SOABI)
install(TARGETS cvl DESTINATION ${SKBUILD_PROJECT_NAME})
set_property(TARGET cvl PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)


target_compile_definitions(cvl PRIVATE CVL_ASSERTS)

if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()
target_include_directories(cvl PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS} src/core/opencl)
if (OpenCL_FOUND)
    target_include_directories(cvl PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
endif ()

add_subdirectory(cpyutl)

target_link_libraries(cvl PRIVATE OpenMP::OpenMP_C ${Python_LIBRARIES} cpyutl)
if (OpenCL_FOUND)
    target_link_libraries(cvl PRIVATE OpenCL::OpenCL)
endif ()

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cvl PRIVATE -Wall -Wextra -Werror -fno-omit-frame-pointer)
elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    # Don't set any warning tags for Windows, since I don't trust it as much as GCC.
    target_compile_options(cvl PRIVATE -openmp:llvm /wd4068 /std:c17)
endif ()


enable_testing()
add_subdirectory(test)
