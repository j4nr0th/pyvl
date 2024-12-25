cmake_minimum_required(VERSION 3.29)
project(cvl_tests C)

enable_testing()

list(APPEND TEST_COMMON_FILES
    test_common.c
    test_common.h
)

add_executable(test_mesh_io mesh_io/test_mesh_io.c ${TEST_COMMON_FILES})
target_link_libraries(test_mesh_io PRIVATE static_cvl m)
add_test(NAME test_mesh_io COMMAND
        test_mesh_io "${CMAKE_SOURCE_DIR}/test/mesh_io/test_mesh.msh")

add_executable(test_dual dual/test_dual.c ${TEST_COMMON_FILES})
target_link_libraries(test_dual PRIVATE static_cvl m)
add_test(NAME test_dual COMMAND
        test_dual "${CMAKE_SOURCE_DIR}/test/dual/test_mesh.msh" "${CMAKE_SOURCE_DIR}/test/dual/test_dual.msh")

add_executable(test_induction induction/test_induction.c ${TEST_COMMON_FILES})
target_link_libraries(test_induction PRIVATE static_cvl m)
add_test(NAME test_induction COMMAND test_induction)

add_executable(test_from_points from_points/test_from_points.c ${TEST_COMMON_FILES})
target_link_libraries(test_from_points PRIVATE static_cvl m)
add_test(NAME test_from_points COMMAND
        test_from_points "${CMAKE_SOURCE_DIR}/test/from_points/test_points.dat"
        "${CMAKE_SOURCE_DIR}/test/from_points/test_mesh.msh")

add_executable(test_to_points from_points/test_to_points.c ${TEST_COMMON_FILES})
target_link_libraries(test_to_points PRIVATE static_cvl m)
add_test(NAME test_to_points COMMAND
        test_to_points "${CMAKE_SOURCE_DIR}/test/from_points/test_points.dat"
        "${CMAKE_SOURCE_DIR}/test/from_points/test_mesh.msh")
