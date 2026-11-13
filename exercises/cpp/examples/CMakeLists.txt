# Every folder here with a main.cpp is one program, built as build/examples/<folder>.exe.
# (The target is ex_<folder>, so the names never clash with legacy's.)
file(GLOB_RECURSE example_mains CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*/main.cpp)
foreach(main ${example_mains})
    get_filename_component(folder ${main} DIRECTORY)
    get_filename_component(name ${folder} NAME)
    add_executable(ex_${name} ${main})
    target_link_libraries(ex_${name} PRIVATE spotter spotter_console_guard)   # the guard: D49
    target_compile_definitions(ex_${name} PRIVATE SPOTTER_REPO_ROOT="${repo_root}")
    set_target_properties(ex_${name} PROPERTIES OUTPUT_NAME ${name} RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)
endforeach()
