add_library(work-inter-${G_APP_SELECT} INTERFACE)
target_include_directories(work-inter-${G_APP_SELECT} INTERFACE ${G_APP_SELECT})

if (NOT EXISTS ${CMAKE_BINARY_DIR}/work_in-${G_APP_SELECT}.toml)
    configure_file(${G_APP_SELECT}/work.toml ${CMAKE_BINARY_DIR}/work_in-${G_APP_SELECT}.toml)
endif()
