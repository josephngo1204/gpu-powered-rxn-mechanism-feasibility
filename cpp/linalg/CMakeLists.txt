add_library(saddle_linalg INTERFACE)
add_library(saddle::linalg ALIAS saddle_linalg)
target_include_directories(saddle_linalg INTERFACE ${PROJECT_SOURCE_DIR}/cpp)
target_link_libraries(saddle_linalg INTERFACE Eigen3::Eigen)
