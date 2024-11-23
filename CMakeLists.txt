cmake_minimum_required(VERSION 3.29)
project(cdust C)

set(CMAKE_C_STANDARD 23)

find_package(OpenMP REQUIRED)

list(
        APPEND CDUST_SOURCE_FILES
        src/core/solver_state.c
        src/core/common.c
        src/core/mesh.c
        src/core/mesh_io.c
        src/core/flow_solver.c
)

list(
        APPEND CDUST_HEADER_FILES
        src/core/solver_state.h
        src/core/common.h
        src/core/mesh.h
        src/core/mesh_io.h
        src/core/flow_solver.h
)

add_library(cdust
        ${CDUST_SOURCE_FILES}
        ${CDUST_HEADER_FILES}
)
target_link_directories(cdust PRIVATE m GOMP)
target_compile_options(cdust PRIVATE -Wall -Wextra -Werror)

enable_testing()
add_subdirectory(test)
