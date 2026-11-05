add_library(saddle_search STATIC rfo.cpp)
add_library(saddle::search ALIAS saddle_search)
target_link_libraries(saddle_search PUBLIC saddle::hess saddle::linalg PRIVATE saddle_warnings)
set_target_properties(saddle_search PROPERTIES POSITION_INDEPENDENT_CODE ON)
