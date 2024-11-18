cmake_minimum_required(VERSION 3.29)
project(cdust_tests C)

enable_testing()

list(APPEND TEST_COMMON_FILES
    test_common.c
    test_common.h
)

add_executable(test_mesh_io mesh_io/test_mesh_io.c ${TEST_COMMON_FILES})
target_link_libraries(test_mesh_io PRIVATE cdust m)
add_test(NAME test_mesh_io COMMAND
        test_mesh_io "${CMAKE_SOURCE_DIR}/test/mesh_io/test_mesh.msh")

add_executable(test_dual dual/test_dual.c ${TEST_COMMON_FILES})
target_link_libraries(test_dual PRIVATE cdust m)
add_test(NAME test_dual COMMAND
        test_dual "${CMAKE_SOURCE_DIR}/test/dual/test_mesh.msh")

