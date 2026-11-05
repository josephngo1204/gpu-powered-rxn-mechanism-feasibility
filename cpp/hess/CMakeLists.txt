add_library(saddle_hess STATIC fd_hessian.cpp noise.cpp)
add_library(saddle::hess ALIAS saddle_hess)
target_link_libraries(saddle_hess PUBLIC saddle::linalg PRIVATE saddle_warnings)
set_target_properties(saddle_hess PROPERTIES POSITION_INDEPENDENT_CODE ON)
