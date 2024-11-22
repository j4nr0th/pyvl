cmake_minimum_required(VERSION 3.29)
project(cdust C)

set(CMAKE_C_STANDARD 23)

list(
        APPEND CDUST_SOURCE_FILES
        src/solver_state.c
        src/common.c
        src/mesh.c
        src/io/mesh_io.c
        src/flow_solver.c
)

list(
        APPEND CDUST_HEADER_FILES
        src/solver_state.h
        src/common.h
        src/mesh.h
        src/io/mesh_io.h
        src/flow_solver.h
)

add_library(cdust
        ${CDUST_SOURCE_FILES}
        ${CDUST_HEADER_FILES}
)
target_link_directories(cdust PRIVATE m)

enable_testing()
add_subdirectory(test)
