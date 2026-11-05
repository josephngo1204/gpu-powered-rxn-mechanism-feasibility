find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC module.cpp)
target_link_libraries(_core PRIVATE saddle::surfaces saddle::hess)

nanobind_add_stub(_core_stub MODULE _core OUTPUT _core.pyi
                  PYTHON_PATH $<TARGET_FILE_DIR:_core> DEPENDS _core)

install(TARGETS _core LIBRARY DESTINATION saddle)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_core.pyi DESTINATION saddle)
