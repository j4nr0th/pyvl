cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "cdust")
endif ()

project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

set(CMAKE_C_STANDARD 23)

find_package(OpenMP REQUIRED)
find_package(OpenACC REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

# Find the NumPy headers
execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(
        APPEND CDUST_MODULE_SOURCE_FILES
        src/module.c
        src/allocator.c
        src/geoidobject.c
        src/lineobject.c
        src/surfaceobject.c
        src/meshobject.c
        src/referenceframeobject.c
)

list(
        APPEND CDUST_MODULE_HEADER_FILES
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
        APPEND CDUST_CORE_SOURCE_FILES
        src/core/solver_state.c
        src/core/common.c
        src/core/mesh.c
        src/core/mesh_io.c
        src/core/flow_solver.c
        src/core/transformation.c
)

list(
        APPEND CDUST_CORE_HEADER_FILES
        src/core/solver_state.h
        src/core/common.h
        src/core/mesh.h
        src/core/mesh_io.h
        src/core/flow_solver.h
        src/core/transformation.h
)

add_library(static_cdust
        ${CDUST_CORE_SOURCE_FILES}
        ${CDUST_CORE_HEADER_FILES}
)

Python_add_library(cdust MODULE ${CDUST_CORE_SOURCE_FILES} ${CDUST_CORE_HEADER_FILES} ${CDUST_MODULE_SOURCE_FILES}
        ${CDUST_MODULE_HEADER_FILES} WITH_SOABI)
install(TARGETS cdust DESTINATION ${SKBUILD_PROJECT_NAME})


target_compile_definitions(cdust PRIVATE CDUST_ASSERTS)

if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message( FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()
target_include_directories(cdust PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

target_compile_options(cdust PRIVATE ${OpenMP_C_FLAGS})
target_link_libraries(cdust PRIVATE ${OpenMP_C_LIBRARIES})

target_compile_options(cdust PRIVATE ${OpenACC_C_FLAGS})
target_link_libraries(cdust PRIVATE ${OpenACC_C_LIBRARIES})


if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cdust PRIVATE -Wall -Wextra -Werror -lto=auto -g -fopt-info-optimized-omp -fno-omit-frame-pointer)
endif ()

#get_target_property(MAIN_CFLAGS cdust COMPILE_OPTIONS)
#message("Compile options are: ${MAIN_CFLAGS}")

enable_testing()
add_subdirectory(test)
