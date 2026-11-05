include(GoogleTest)

function(saddle_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE saddle::linalg saddle_warnings
                        GTest::gtest_main rapidcheck rapidcheck_gtest)
  gtest_discover_tests(${name} PROPERTIES LABELS unit)
endfunction()

saddle_add_test(test_smoke test_smoke.cpp)
saddle_add_test(test_linalg test_linalg.cpp)
saddle_add_test(test_surfaces test_surfaces.cpp)
target_link_libraries(test_surfaces PRIVATE saddle::surfaces)
saddle_add_test(test_fd_hessian test_fd_hessian.cpp)
target_link_libraries(test_fd_hessian PRIVATE saddle::hess saddle::surfaces)
