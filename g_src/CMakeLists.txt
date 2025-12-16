add_library(src-inter-${G_APP_SELECT} INTERFACE)
target_include_directories(src-inter-${G_APP_SELECT} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
target_sources(src-inter-${G_APP_SELECT} INTERFACE main.cu)

configure_file(CMakeHeader.h.in CMakeHeader.h @ONLY)
if (NOT EXISTS ${CMAKE_BINARY_DIR}/options_in.toml)
    configure_file(options.toml ${CMAKE_BINARY_DIR}/options_in.toml)
endif()
