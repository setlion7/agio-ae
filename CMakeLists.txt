cmake_minimum_required(VERSION 3.31)
project(ideal-potato VERSION 1.0 LANGUAGES CUDA CXX C)

# Set language options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
    # add_compile_options(-g -G)
endif()

# Set work
set(G_APP_SELECT_LIST)
list(APPEND G_APP_SELECT_LIST gemv)
list(APPEND G_APP_SELECT_LIST kmeans)
list(APPEND G_APP_SELECT_LIST bfs)
list(APPEND G_APP_SELECT_LIST sssp)
list(APPEND G_APP_SELECT_LIST pagerank)
list(APPEND G_APP_SELECT_LIST cc)

# Other flags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CUDA_ARCHITECTURES native)
add_compile_options(-Wall)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g")
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo --source-in-ptx --keep --resource-usage")

add_subdirectory(subtree/bam)
add_subdirectory(g_nvme)

foreach(G_APP_SELECT ${G_APP_SELECT_LIST})
    message("Configure work: ${G_APP_SELECT}")
    add_executable(main-${G_APP_SELECT})

    OPTION(G_DEBUG "Set G_DEBUG" OFF)
    if (G_DEBUG)
        target_compile_definitions(main-${G_APP_SELECT} PUBLIC -DG_DEBUG)
    endif()
    # target_compile_options(main-${G_APP_SELECT} PRIVATE PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -g>)

    # Subtrees
    target_link_libraries(main-${G_APP_SELECT} libnvm)
    target_include_directories(main-${G_APP_SELECT} PRIVATE subtree/bam/include)
    target_include_directories(main-${G_APP_SELECT} PRIVATE subtree/bam/include/freestanding/include)

    target_include_directories(main-${G_APP_SELECT} PRIVATE g_nvme)
    target_link_libraries(main-${G_APP_SELECT} g_nvme)

    add_subdirectory(g_addon gaddon-${G_APP_SELECT})
    add_subdirectory(g_io gio-${G_APP_SELECT})
    target_link_libraries(main-${G_APP_SELECT} guser-inter-${G_APP_SELECT})
    target_link_libraries(main-${G_APP_SELECT} gio-inter-${G_APP_SELECT})

    add_subdirectory(app work-${G_APP_SELECT})
    target_link_libraries(main-${G_APP_SELECT} work-inter-${G_APP_SELECT})

    add_subdirectory(g_src gsrc-${G_APP_SELECT})
    target_link_libraries(main-${G_APP_SELECT} src-inter-${G_APP_SELECT})

    target_link_libraries(main-${G_APP_SELECT} CUDA::cuda_driver CUDA::nvml CUDA::nvtx3)
endforeach()

message("Done.")
