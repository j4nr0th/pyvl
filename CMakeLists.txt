cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "cvl")
endif ()

project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

set(CMAKE_C_STANDARD 17)

find_package(OpenMP REQUIRED)
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
        src/core/barnes_hut_tree.c
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
        src/core/barnes_hut_tree.h
)

add_library(static_cvl
        ${CVL_CORE_SOURCE_FILES}
        ${CVL_CORE_HEADER_FILES}
        src/core/multipole.c
)
set_property(TARGET static_cvl PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
target_link_libraries(static_cvl PUBLIC OpenMP::OpenMP_C)


Python_add_library(cvl MODULE ${CVL_CORE_SOURCE_FILES} ${CVL_CORE_HEADER_FILES} ${CVL_MODULE_SOURCE_FILES}
        ${CVL_MODULE_HEADER_FILES} WITH_SOABI)
install(TARGETS cvl DESTINATION ${SKBUILD_PROJECT_NAME})
set_property(TARGET cvl PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)


target_compile_definitions(cvl PRIVATE CVL_ASSERTS)

if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()
target_include_directories(cvl PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

add_subdirectory(cpyutl)

target_link_libraries(cvl PRIVATE OpenMP::OpenMP_C ${Python_LIBRARIES} cpyutl)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cvl PRIVATE -Wall -Wextra -Werror -fno-omit-frame-pointer)
elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    # Don't set any warning tags for Windows, since I don't trust it as much as GCC.
    target_compile_options(cvl PRIVATE -openmp:llvm /wd4068 /std:c17)
endif ()


enable_testing()
add_subdirectory(test)
