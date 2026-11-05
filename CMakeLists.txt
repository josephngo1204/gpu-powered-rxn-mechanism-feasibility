cmake_minimum_required(VERSION 3.28)
project(saddle VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

option(SADDLE_ENABLE_CUDA "Build the cuEST adapter and GPU code" OFF)
option(SADDLE_ENABLE_SANITIZERS "Build with ASan and UBSan" OFF)
option(SADDLE_BUILD_TESTS "Build C++ tests" ON)
option(SADDLE_BUILD_PYTHON "Build the nanobind module" OFF)

add_library(saddle_warnings INTERFACE)
target_compile_options(saddle_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror)

if(SADDLE_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(Eigen3 3.4...5 REQUIRED NO_MODULE)

add_subdirectory(cpp)

if(SADDLE_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  find_package(rapidcheck REQUIRED)
  add_subdirectory(tests/cpp)
endif()
