include(GoogleTest)

function(saddle_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE saddle::linalg saddle_warnings
                        GTest::gtest_main rapidcheck rapidcheck_gtest)
  gtest_discover_tests(${name} PROPERTIES LABELS unit)
endfunction()

saddle_add_test(test_smoke test_smoke.cpp)
