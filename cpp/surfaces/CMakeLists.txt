add_library(saddle_surfaces STATIC surfaces.cpp)
add_library(saddle::surfaces ALIAS saddle_surfaces)
target_link_libraries(saddle_surfaces PUBLIC saddle::linalg PRIVATE saddle_warnings)
set_target_properties(saddle_surfaces PROPERTIES POSITION_INDEPENDENT_CODE ON)
