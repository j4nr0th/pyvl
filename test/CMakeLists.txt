cmake_minimum_required(VERSION 3.29)
project(cvl_tests C)

enable_testing()

list(APPEND TEST_COMMON_FILES
    test_common.c
    test_common.h
)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(TEST_MATH_LIB m)
endif ()

add_executable(test_mesh_io mesh_io/test_mesh_io.c ${TEST_COMMON_FILES})
target_link_libraries(test_mesh_io PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_mesh_io COMMAND
        test_mesh_io "${CMAKE_SOURCE_DIR}/test/mesh_io/test_mesh.msh")

add_executable(test_dual dual/test_dual.c ${TEST_COMMON_FILES})
target_link_libraries(test_dual PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_dual COMMAND
        test_dual "${CMAKE_SOURCE_DIR}/test/dual/test_mesh.msh" "${CMAKE_SOURCE_DIR}/test/dual/test_dual.msh")

add_executable(test_induction induction/test_induction.c ${TEST_COMMON_FILES})
target_link_libraries(test_induction PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_induction COMMAND test_induction)

add_executable(test_from_points from_points/test_from_points.c ${TEST_COMMON_FILES})
target_link_libraries(test_from_points PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_from_points COMMAND
        test_from_points "${CMAKE_SOURCE_DIR}/test/from_points/test_points.dat"
        "${CMAKE_SOURCE_DIR}/test/from_points/test_mesh.msh")

add_executable(test_to_points from_points/test_to_points.c ${TEST_COMMON_FILES})
target_link_libraries(test_to_points PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_to_points COMMAND
        test_to_points "${CMAKE_SOURCE_DIR}/test/from_points/test_points.dat"
        "${CMAKE_SOURCE_DIR}/test/from_points/test_mesh.msh")

add_executable(test_approx_atan math/test_approx_atan.c ${TEST_COMMON_FILES})
target_link_libraries(test_approx_atan PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_approx_atan COMMAND test_approx_atan "1000")

add_executable(test_approx_atan2 math/test_approx_atan2.c ${TEST_COMMON_FILES})
target_link_libraries(test_approx_atan2 PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_approx_atan2 COMMAND test_approx_atan2 "1000")

add_executable(test_cost_model math/test_cost_model.c ${TEST_COMMON_FILES})
target_link_libraries(test_cost_model PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_cost_model COMMAND test_cost_model)

add_executable(test_multipole math/test_multipole.c ${TEST_COMMON_FILES})
target_link_libraries(test_multipole PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_multipole COMMAND test_multipole)

add_executable(test_multipole_shift math/test_multipole_shift.c ${TEST_COMMON_FILES})
target_link_libraries(test_multipole_shift PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_multipole_shift COMMAND test_multipole_shift)

add_executable(test_barnes_hut math/test_barnes_hut.c ${TEST_COMMON_FILES})
target_link_libraries(test_barnes_hut PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_barnes_hut COMMAND test_barnes_hut)

add_executable(test_barnes_hut_perf math/test_barnes_hut_perf.c ${TEST_COMMON_FILES})
target_link_libraries(test_barnes_hut_perf PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_barnes_hut_perf COMMAND test_barnes_hut_perf)
target_compile_options(test_barnes_hut_perf PRIVATE -O3 -flto=auto)
set_property(TARGET test_barnes_hut_perf PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_tests_properties(test_barnes_hut_perf PROPERTIES TIMEOUT 120)

add_executable(test_bh_accuracy math/test_bh_accuracy.c ${TEST_COMMON_FILES})
target_link_libraries(test_bh_accuracy PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_bh_accuracy COMMAND test_bh_accuracy)
target_compile_options(test_bh_accuracy PRIVATE -O3 -flto=auto)
set_property(TARGET test_bh_accuracy PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_tests_properties(test_bh_accuracy PROPERTIES TIMEOUT 180)

add_executable(test_fmm_operators math/test_fmm_operators.c ${TEST_COMMON_FILES})
target_link_libraries(test_fmm_operators PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_fmm_operators COMMAND test_fmm_operators)

add_executable(test_fmm_tree math/test_fmm_tree.c ${TEST_COMMON_FILES})
target_link_libraries(test_fmm_tree PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_fmm_tree COMMAND test_fmm_tree)

add_executable(test_fmm_accuracy math/test_fmm_accuracy.c ${TEST_COMMON_FILES})
target_link_libraries(test_fmm_accuracy PRIVATE static_cvl ${TEST_MATH_LIB})
add_test(NAME test_fmm_accuracy COMMAND test_fmm_accuracy)
set_tests_properties(test_fmm_accuracy PROPERTIES TIMEOUT 120)
