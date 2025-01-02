cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "cvl")
endif ()

project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

set(CMAKE_C_STANDARD 17)

find_package(OpenMP REQUIRED)
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
        src/lineobject.c
        src/surfaceobject.c
        src/meshobject.c
        src/referenceframeobject.c
)

list(
        APPEND CVL_MODULE_HEADER_FILES
        src/module.h
        src/allocator.h
        src/geoidobject.h
        src/lineobject.h
        src/surfaceobject.h
        src/meshobject.h
        src/referenceframeobject.h
        src/common.h
)

list(
        APPEND CVL_CORE_SOURCE_FILES
        src/core/solver_state.c
        src/core/common.c
        src/core/mesh.c
        src/core/mesh_io.c
        src/core/flow_solver.c
        src/core/transformation.c
)

list(
        APPEND CVL_CORE_HEADER_FILES
        src/core/solver_state.h
        src/core/common.h
        src/core/mesh.h
        src/core/mesh_io.h
        src/core/flow_solver.h
        src/core/transformation.h
)

add_library(static_cvl
        ${CVL_CORE_SOURCE_FILES}
        ${CVL_CORE_HEADER_FILES}
)

Python_add_library(cvl MODULE ${CVL_CORE_SOURCE_FILES} ${CVL_CORE_HEADER_FILES} ${CVL_MODULE_SOURCE_FILES}
        ${CVL_MODULE_HEADER_FILES} WITH_SOABI)
install(TARGETS cvl DESTINATION ${SKBUILD_PROJECT_NAME})


target_compile_definitions(cvl PRIVATE CVL_ASSERTS)

if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message( FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()
target_include_directories(cvl PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})


target_link_libraries(cvl PRIVATE ${OpenMP_C_LIBRARIES} ${Python_LIBRARIES})

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cvl PRIVATE ${OpenMP_C_FLAGS})
    target_compile_options(cvl PRIVATE -Wall -Wextra -Werror -lto=auto -std=c17 -fno-omit-frame-pointer)
elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    # Don't set any warning tags for Windows, since I don't trust it as much as GCC.
    target_compile_options(cvl PRIVATE -openmp:llvm /wd4068 /std:c17)
endif ()


enable_testing()
add_subdirectory(test)
